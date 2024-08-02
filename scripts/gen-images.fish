#!/usr/bin/env fish

set ninja build.ninja

echo 'imflags = -limit thread 1 -quality 70' > $ninja
echo 'rule im' >> $ninja
echo '  command = convert $in $imflags $out' >> $ninja
echo 'rule im-1080' >> $ninja
echo '  command = convert $in $imflags -resize 1920x1080 $out' >> $ninja
echo 'rule im-720' >> $ninja
echo '  command = convert $in $imflags -resize 1280x720 $out' >> $ninja
# echo 'rule im-450' >> $ninja
# echo '  command = convert $in $imflags -resize 1280x720 $out' >> $ninja
echo '' >> $ninja

# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-1440.avif'
# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-1440.webp'
# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -resize 1920x1080 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-1080.avif'
# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -resize 1280x720 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-720.avif'
# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -resize 800x450 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-450.avif'
# find img/ | rg  '\.png$' | rg -v 'pngdir' | parallel convert '{}' -quality 70 -resize 800x450 -set filename:fn '%d/%t' +adjoin '%[filename:fn]-450.webp'


find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-1440.avif: im \1.png/' >> $ninja
# find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-1440.webp: im \1.png/' >> $ninja
find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-1080.avif: im-1080 \1.png/' >> $ninja
find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-720.avif: im-720 \1.png/' >> $ninja
# find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-450.avif: im-450 \1.png/' >> $ninja
# find img/ | rg  '\.png$' | rg -v 'pngdir' | sed -E 's/(.+)[.]png/build \1-450.webp: im-450 \1.png/' >> $ninja

