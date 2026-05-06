import torch
from util import get_text_data
import json



def generate_ood_id_dataset(ood_data, id_data, ood_id_positive_edges):
    
    dataset = []
    for i, lst in enumerate(ood_id_positive_edges):
        ood_sample = ood_data[i]
        id_sample = id_data[lst[0]]
        prompt = "Based on the existing label sapce and the title and abstract of a new paper. Does it belong to one of the existing labels, or does it not? \n\nLable sapce: Label sapce: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning or Probabilistic_Methods. "
        data1 = {"id": f"{i*2}",
                "conversations": [
                    {   "from": "human",
                        "value": prompt + f'\n\nThe paper:\n{ood_sample}',
                    },
                    {
                        "from": "gpt",
                        "value": "False"
                    }
                ]
                }
        
        data2 = {"id": f"{i*2+1}",
                "conversations": [
                    {   "from": "human",
                        "value": prompt + f'\n\nThe paper:\n{id_sample}',
                    },
                    {
                        "from": "gpt",
                        "value": "True"
                    }
                ]
                }
        dataset.append(data1)
        dataset.append(data2)
        # if i > 5:
        #     break
        # import ipdb; ipdb.set_trace()
    # save dataset to json file
    with open("cora_node_type.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    

if __name__ == '__main__':
    sood_data, id_data = get_text_data(dataname='cora')
    generate_ood_id_dataset(ood_data, id_data, ood_id_positive_edges)
                                  
                                  