for file in $(ls ./ | grep -v -E 'png|bash' | cut -d. -f 1); do echo $file; pdftoppm ${file}.pdf ${file} -png -scale-to 3508; done
