for file in .build_release/tools/*.bin; do
    cp "$file" "${file%.bin}"
done
