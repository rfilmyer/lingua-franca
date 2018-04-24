for file in ~/Downloads/italian-zip/*.tgz; do 
	tar -xzf $file -C zips/italian/; #errors with spaces in the name (intended behavior, as a quick way of deduping)
done