SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
MATH=wolfram-12.0.0
$MATH -noprompt -script $SCRIPTPATH/render3d.m "${@:1}"
convert -delay 30 out*.png  -loop 0 test.gif
rm out*.png
