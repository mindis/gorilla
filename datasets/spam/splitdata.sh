split -l 500 web*.ds
i=1
for x in `ls x* | sort`
do
    mv $x part-$i.ds
    i=$(($i+1))
done
