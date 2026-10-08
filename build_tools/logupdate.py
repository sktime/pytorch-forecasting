import argparse
import io
import sys

# CLI argument parsing
parser = argparse.ArgumentParser(description="changelog from merged PRs")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to export changelog as Markdown"
)
args = parser.parse_args()

# Capture output
buffer = io.StringIO()
original_stdout = sys.stdout
sys.stdout = buffer

# Render changelog and contributors 
render_changelog(pulls, assigned)
render_contributors(pulls)

# Restore stdout
sys.stdout = original_stdout
changelog_content = buffer.getvalue()

# Write to file if requested 
if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(changelog_content)
    print(f"Changelog written to {args.output}")
else:
    # If no output file, just print to terminal
    print(changelog_content)
