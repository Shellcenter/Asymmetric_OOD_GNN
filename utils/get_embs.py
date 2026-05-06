import torch
import numpy as np
from tqdm import tqdm


from transformers import AutoTokenizer
from CustomVicuna import CustomVicunaForCausalLM
from get_new_edges import get_ood_id_links

path = "./saved_model/cora/vicuna_tuned"
device = "cuda:4"

tokenizer = AutoTokenizer.from_pretrained(path)
model = CustomVicunaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).to(device)


@torch.no_grad()
def get_emb(texts, model, tokenizer):
    embs = []
    for text in tqdm(texts):
        inputs = tokenizer(text)
        input_ids = torch.as_tensor(inputs.input_ids).unsqueeze(0).to(device)
        attention_mask = torch.as_tensor(inputs.attention_mask).unsqueeze(0).to(device)
        out = model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        # import ipdb; ipdb.set_trace()
        emb = out['hidden_states'][-1].squeeze().mean(0)
        embs.append(emb.cpu().detach().numpy())
    return embs

if __name__ == "__main__":
    ood_data, id_data, _, _ = get_ood_id_links(dataname='cora', threshold=3)
    
    ood_embs = get_emb(ood_data, model, tokenizer)
    # import ipdb; ipdb.set_trace()
    np.save('cora/cora_ood_embs', np.array(ood_embs))
    
    id_embs = get_emb(id_data, model, tokenizer)
    np.save('cora/cora_id_embs', np.array(id_embs))