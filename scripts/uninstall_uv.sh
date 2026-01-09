# clean stored data
uv cache clean
rm -r "$(uv python dir)"
rm -r "$(uv tool dir)"
# remove binaries
rm ~/.local/bin/uv ~/.local/bin/uvx
