echo 'REFOCUSING PULSE DESIGN'

python demo_refocusing.py --b0map 0 --b1map 0 --folder refocusing_00 --optimization 1
python demo_refocusing.py --b0map 0 --b1map 0 --folder refocusing_00 --evaluation 1
# --- and simulate with differen b0 and b1 maps
python demo_refocusing.py --b0map 1 --b1map 0 --folder refocusing_00 --evaluation 1 
python demo_refocusing.py --b0map 2 --b1map 0 --folder refocusing_00 --evaluation 1 
python demo_refocusing.py --b0map 3 --b1map 0 --folder refocusing_00 --evaluation 1 
python demo_refocusing.py --b0map 4 --b1map 0 --folder refocusing_00 --evaluation 1 
python demo_refocusing.py --b0map 0 --b1map 1 --folder refocusing_00 --evaluation 1 
python demo_refocusing.py --b0map 0 --b1map 2 --folder refocusing_00 --evaluation 1
python demo_refocusing.py --b0map 0 --b1map 3 --folder refocusing_00 --evaluation 1


# --- tailored designs
python demo_refocusing.py --b0map 1 --b1map 0 --folder refocusing_10 --optimization 1 
python demo_refocusing.py --b0map 1 --b1map 0 --folder refocusing_10 --evaluation 1

python demo_refocusing.py --b0map 2 --b1map 0 --folder refocusing_20 --optimization 1
python demo_refocusing.py --b0map 2 --b1map 0 --folder refocusing_20 --evaluation 1

python demo_refocusing.py --b0map 3 --b1map 0 --folder refocusing_30 --optimization 1 
python demo_refocusing.py --b0map 3 --b1map 0 --folder refocusing_30 --evaluation 1

python demo_refocusing.py --b0map 4 --b1map 0 --folder refocusing_40 --optimization 1 
python demo_refocusing.py --b0map 4 --b1map 0 --folder refocusing_40 --evaluation 1

python demo_refocusing.py --b0map 0 --b1map 1 --folder refocusing_01 --optimization 1 
python demo_refocusing.py --b0map 0 --b1map 1 --folder refocusing_01 --evaluation 1

python demo_refocusing.py --b0map 0 --b1map 2 --folder refocusing_02 --optimization 1 
python demo_refocusing.py --b0map 0 --b1map 2 --folder refocusing_02 --evaluation 1

python demo_refocusing.py --b0map 0 --b1map 3 --folder refocusing_03 --optimization 1 
python demo_refocusing.py --b0map 0 --b1map 3 --folder refocusing_03 --evaluation 1



echo 'EXCITATION PULSE DESIGN'

python demo_excitation.py --b0map 0 --b1map 0 --folder excitation_00 --optimization 1
python demo_excitation.py --b0map 0 --b1map 0 --folder excitation_00 --evaluation 1
# --- and simulate with differen b0 and b1 maps
python demo_excitation.py --b0map 1 --b1map 0 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 2 --b1map 0 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 3 --b1map 0 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 4 --b1map 0 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 0 --b1map 1 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 0 --b1map 2 --folder excitation_00 --evaluation 1 
python demo_excitation.py --b0map 0 --b1map 3 --folder excitation_00 --evaluation 1

# --- tailored designs
python demo_excitation.py --b0map 1 --b1map 0 --folder excitation_10 --optimization 1 
python demo_excitation.py --b0map 1 --b1map 0 --folder excitation_10 --evaluation 1

python demo_excitation.py --b0map 2 --b1map 0 --folder excitation_20 --optimization 1 
python demo_excitation.py --b0map 2 --b1map 0 --folder excitation_20 --evaluation 1

python demo_excitation.py --b0map 3 --b1map 0 --folder excitation_30 --optimization 1
python demo_excitation.py --b0map 3 --b1map 0 --folder excitation_30 --evaluation 1

python demo_excitation.py --b0map 4 --b1map 0 --folder excitation_40 --optimization 1
python demo_excitation.py --b0map 4 --b1map 0 --folder excitation_40 --evaluation 1

python demo_excitation.py --b0map 0 --b1map 1 --folder excitation_01 --optimization 1
python demo_excitation.py --b0map 0 --b1map 1 --folder excitation_01 --evaluation 1

python demo_excitation.py --b0map 0 --b1map 2 --folder excitation_02 --optimization 1 
python demo_excitation.py --b0map 0 --b1map 2 --folder excitation_02 --evaluation 1

python demo_excitation.py --b0map 0 --b1map 3 --folder excitation_03 --optimization 1 
python demo_excitation.py --b0map 0 --b1map 3 --folder excitation_03 --evaluation 1
