# Transclude sections of the README into the docs pages.
module DocHelpers
using Markdown
const README = read(joinpath(@__DIR__, "..", "README.md"), String)

function section_level(s)
    count = 0
    for c in strip(s)
        if c == '#'
            count += 1
        else
            break
        end
    end
    return count
end

function readme_section(target_header::String)
    if !startswith(strip(target_header), "#")
        throw(ArgumentError("target_header must start with a #"))
    end

    header_level = section_level(target_header)

    # Split the README into lines for processing
    lines = split(README, '\n')

    inside_code_block = false
    capturing = false
    captured_istart = 1
    captured_iend = 1

    for i in axes(lines, 1)
        line = strip(lines[i])

        # Check for the start or end of a code block
        if startswith(line, "```")
            inside_code_block = !inside_code_block
            continue
        end

        if !inside_code_block
            # Check if the current line is the target header
            if !capturing && line == target_header
                capturing = true
                captured_istart = i + 1
                continue  # Skip adding the header itself
            end

            # If capturing, check for the next header of same or higher level to stop
            if capturing && startswith(line, "#")
                current_level = section_level(line)
                if current_level <= header_level
                    captured_iend = i - 1
                    break  # Stop capturing as we've reached the next section
                end
            end
        end
    end

    if !capturing
        throw(ArgumentError("Could not find section $target_header in README"))
    end

    # Join the captured lines back into a single string
    return Markdown.parse(join(lines[captured_istart:captured_iend], '\n'))
end
end
