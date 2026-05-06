### cora
device=5
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -4 --device $device
