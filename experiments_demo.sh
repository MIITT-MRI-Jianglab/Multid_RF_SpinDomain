echo 'REFOCUSING PULSE DESIGN'
python demo_refocusing.py --b0map 1 --b1map 1 --folder refocusing_demo --optimization 1
python demo_refocusing.py --b0map 1 --b1map 1 --folder refocusing_demo --evaluation 1


echo 'EXCITATION PULSE DESIGN'
python demo_excitation.py --b0map 1 --b1map 1 --folder excitation_demo --optimization 1
python demo_excitation.py --b0map 1 --b1map 1 --folder excitation_demo --evaluation 1
