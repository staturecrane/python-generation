for dir in */; do
  while read -r f; do
    cat "$f" >> allpy.txt
    echo "--------" >> allpy.txt
  done < <(find "$dir" -type f -name "*.py")
done
