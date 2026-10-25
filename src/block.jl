# Block-level device primitives shared by the algorithm families: reductions and scans of a
# block's values, and vector loads.

# Tree-reduce the `@groupsize()[1]` values in `sdata` into `sdata[1]`; the group size must be a power
# of two. All threads in the group must call it, after writing their value and synchronising.
@inline function reduce_group!(@context, op, sdata, ithread)
    # The group size is static, so this loop has a constant trip count; the conversion keeps the
    # index arithmetic in `ithread`'s integer type.
    s = typeof(ithread)(@groupsize()[1] >> 1)
    while s > 0x0
        if ithread < s
            sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + s + 0x1])
        end
        @synchronize()
        s >>= 0x1
    end
end


# The sub-group width `W` for a block scan of `T` in blocks of `block_size` threads, as a `Val`,
# or `Val(0)` for the local-memory scan. Sub-groups must be enabled by the scan tuning, linear
# (from KernelAbstractions 0.10), able to shuffle `T`, and `W ≤ block_size ≤ W²`, as one
# sub-group scans the sub-groups' totals.
@static if isdefined(KernelAbstractions, :KernelInterface)
    function scan_subgroup_width(backend, ::Type{T}, block_size) where {T}
        scan_tuning(backend, T).subgroups || return Val(0)
        KI.supports_linear_subgroups(backend) && KI.supports_shuffle(backend, T) || return Val(0)
        W = KI.sub_group_size(backend)
        return ispow2(W) && W <= block_size <= W * W ? Val(W) : Val(0)
    end
else
    scan_subgroup_width(backend, T, block_size) = Val(0)
end


# Exclusive scan of one value per thread, `totals[ithread + 1]`, seeded with `seed`: returns the
# thread's prefix and the block's total. All threads in the block must call it. Operands are
# combined in element order (earlier on the left), so `op` need not be commutative. With a
# sub-group width `W` (see `scan_subgroup_width`), it scans with sub-group operations, else in local
# memory. It overwrites `totals`.
@inline block_exclusive_scan!(@context, op, totals, seed, block_size, ithread, ::Val{0}) =
    block_exclusive_scan!(@context, op, totals, seed, block_size, ithread)

@inline function block_exclusive_scan!(@context, op, totals, seed, block_size, ithread,
                                       ::Val{W}) where {W}
    # scan each sub-group's values, then the sub-groups' totals, in one sub-group
    x = totals[ithread + 0x1]
    lane = ithread % W
    sg = ithread ÷ W
    nsg = block_size ÷ W
    incl = KI.sub_group_scan(op, x)
    @synchronize()
    if lane == W - 1
        totals[sg + 0x1] = incl
    end
    @synchronize()
    # (lanes beyond the sub-groups only affect results that are not used)
    t = lane < nsg ? totals[lane + 0x1] : seed
    sg_excl = KI.sub_group_exclusive_scan(op, t, seed)
    sg_prefix = KI.shfl(sg_excl, sg + 0x1)
    block_total = KI.shfl(op(sg_excl, t), nsg)
    prev = KI.shfl_up(incl, 1)
    @synchronize()
    return (lane == 0 ? sg_prefix : op(sg_prefix, prev)), block_total
end

# The scan in local memory
@inline function block_exclusive_scan!(@context, op, totals, seed, block_size, ithread)
    # Up-sweep
    offset = one(ithread)
    d = block_size >> 0x1
    while d > 0x0
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            totals[bi + 0x1] = op(totals[ai + 0x1], totals[bi + 0x1])
        end
        offset = offset << 0x1
        d = d >> 0x1
    end

    @synchronize()
    block_total = op(seed, totals[block_size])
    @synchronize()
    if ithread == 0x0
        totals[block_size] = seed
    end

    # Down-sweep to an exclusive scan.
    d = one(ithread)
    while d < block_size
        offset = offset >> 0x1
        @synchronize()
        if ithread < d
            ai = offset * (0x2 * ithread + 0x1) - 0x1
            bi = offset * (0x2 * ithread + 0x2) - 0x1
            t = totals[ai + 0x1]
            totals[ai + 0x1] = totals[bi + 0x1]
            totals[bi + 0x1] = op(totals[bi + 0x1], t)
        end
        d = d << 0x1
    end
    @synchronize()

    return totals[ithread + 0x1], block_total
end


# Aligned load of `W` consecutive elements as one LLVM vector (`load <W x T>, align W*sizeof(T)`).
# Julia 1.12 (LLVM 17+) passes `LLVMPtr` as an opaque `ptr`; older versions pass it as `i8*`,
# which needs a bitcast to the vector pointer type.
@generated function vload(::Val{W}, ptr::Core.LLVMPtr{T, AS}) where {W, T, AS}
    lt = T === Float32 ? "float" : T === Float64 ? "double" : "i$(8 * sizeof(T))"
    vt = "<$W x $lt>"
    align = W * sizeof(T)
    ir = @static if VERSION >= v"1.12-"
        """
        %res = load $vt, ptr addrspace($AS) %0, align $align
        ret $vt %res
        """
    else
        """
        %ptr = bitcast i8 addrspace($AS)* %0 to $vt addrspace($AS)*
        %res = load $vt, $vt addrspace($AS)* %ptr, align $align
        ret $vt %res
        """
    end
    return quote
        $(Expr(:meta, :inline))
        Base.llvmcall($ir, NTuple{W, Core.VecElement{T}}, Tuple{Core.LLVMPtr{T, AS}}, ptr)
    end
end

# Fold a vector into the thread-local accumulator. `W` is known at compile time.
@inline fold_vector(f, op, acc, v::NTuple{W, Core.VecElement}) where {W} =
    foldl(op, ntuple(k -> f(@inbounds v[k].value), Val(W)); init = acc)
