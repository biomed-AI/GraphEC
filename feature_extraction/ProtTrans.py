import gc
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel


def get_ProtTrans(ID_list, seq_list, ProtTrans_path, outpath, gpu):
    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
    gc.collect()

    # Load the model into the GPU if avilabile and switch to inference mode
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.to(device)
    model = model.eval()

    print("Extracting ProtTrans embeddings...")
    for i in tqdm(range(len(ID_list))):
        batch_ID_list = [ID_list[i]] # batch size = 1
        batch_seq_list = [" ".join(list(seq_list[i]))]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu()

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            torch.save(seq_emd, outpath + batch_ID_list[seq_num] + '.tensor')
