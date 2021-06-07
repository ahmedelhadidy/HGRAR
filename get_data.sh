expr=".*,$1,.*,$2,.*"
echo $expr
grep -r $expr ./test_data/* | awk -F , -v f1=$1 -v f2=$2  '{
if($7 == f1 && $10 == f2)
{
print $0;
}
}
'