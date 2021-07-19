

get_grars(){
  run="$1"
  usecase="$2"
  cls="$3"
  count=$(($4))
  grar_type=''
  if [[ $cls == "P" ]]; then
    grar_type='faulty_grar'
  else
    grar_type='non_faulty_grar'
  fi
  cat ~/hygrar/$run/$usecase/grars.json | jq -jc " .$grar_type[] | .membership  , \"\t\"  ,.rule_terms[0].operator.NN_model.identifier , \"\n\" " | sort -r | head -$count
}


CLASS=''
COUNT=0
run_id=""
usecase=""
while getopts ":pnc:r:u:" opt; do
  case ${opt} in
    p ) CLASS='P'
      ;;
    n ) CLASS='N'
      ;;
    c ) COUNT=$OPTARG
      ;;
    r ) run_id=$OPTARG
    ;;
    u ) usecase=$OPTARG
    ;;
    \? ) echo "Usage: cmd [-p | -n] -c"
      ;;
  esac
done
shift $((OPTIND -1))
get_grars $run_id $usecase $CLASS $COUNT