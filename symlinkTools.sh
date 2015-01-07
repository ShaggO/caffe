#!/bin/bash
echo "Manual linking of .bin files in .build_release/tools/ to their non-.bin names"
echo "Change directory to .build_release/tools"
cd .build_release/tools/
for i in *.bin; do
    file="${i%.*}"
    if [ -L $file ]; then
        echo "$file exists. Remove it!"
        rm "$file"
    fi
    echo "Symlink $i to $file"
    ln -s "$i" "${file}"
done;
cd ../../
