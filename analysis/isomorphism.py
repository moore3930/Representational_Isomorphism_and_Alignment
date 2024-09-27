import numpy
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch
import torch.nn as nn
import torch.optim as optim


def topK_cos_retrieve(tensor1, tensor2, k):
    AB = numpy.matmul(tensor1, tensor2.T)
    a = np.linalg.norm(tensor1, axis=1).reshape(1, -1)
    b = np.linalg.norm(tensor2, axis=1).reshape(1, -1)
    ab = numpy.matmul(a.T, b)
    cos_matrix = AB / ab

    topk_indices_per_row = np.argsort(cos_matrix, axis=1)[:, -k:]

    # Create a zero-filled tensor with the same shape as cos_matrix
    topk_predict = np.zeros_like(cos_matrix)

    # Set elements within top-k indices to 1
    for i, row in enumerate(topk_indices_per_row):
        topk_predict[i, row] = 1

    top1_gold = np.eye(tensor1.shape[0])
    p = (top1_gold * topk_predict).sum()
    topk_acc = round(float(p) / tensor1.shape[0], 2)

    return topk_acc


def get_acc(src_path, tgt_path, seed, k=10, method="centre"):
    en_emb_path = src_path
    de_emb_path = tgt_path

    torch.manual_seed(seed)

    en_emb = torch.load(en_emb_path)
    de_emb = torch.load(de_emb_path)
    perm_indices = torch.randperm(len(en_emb))
    en_emb = en_emb[perm_indices]
    de_emb = de_emb[perm_indices]

    X_train = en_emb[:k].numpy()
    Y_train = de_emb[:k].numpy()

    X_test = en_emb[k:].numpy()
    Y_test = de_emb[k:].numpy()

    R, _ = orthogonal_procrustes(X_train, Y_train)
    X_proj = np.matmul(X_test, R)

    # format output
    results = {}
    results["P@5-org"] = "{:.2f}".format(topK_cos_retrieve(X_test, Y_test, 5))
    results["P@5-proj"] = "{:.2f}".format(topK_cos_retrieve(X_proj, Y_test, 5))

    return results

# English Prompts
en_emb_path = "./english-prompt/Llama-2-7b-hf-eng_Latn-32bit"
de_emb_path = "./english-prompt/Llama-2-7b-hf-deu_Latn-32bit"
es_emb_path = "./english-prompt/Llama-2-7b-hf-spa_Latn-32bit"
ar_emb_path = "./english-prompt/Llama-2-7b-hf-arb_Arab-32bit"
zh_emb_path = "./english-prompt/Llama-2-7b-hf-zho_Hans-32bit"
jp_emb_path = "./english-prompt/Llama-2-7b-hf-jpn_Jpan-32bit"
ru_emb_path = "./english-prompt/Llama-2-7b-hf-rus_Cyrl-32bit"

# Self-Language Prompts
en_emb_path = "./self-lang-prompt/Llama-2-7b-hf-eng_Latn-32bit"
ar_emb_path = "./self-lang-prompt/Llama-2-7b-hf-arb_Arab-32bit"
zh_emb_path = "./self-lang-prompt/Llama-2-7b-hf-zho_Hans-32bit"
jp_emb_path = "./self-lang-prompt/Llama-2-7b-hf-jpn_Jpan-32bit"
ru_emb_path = "./self-lang-prompt/Llama-2-7b-hf-rus_Cyrl-32bit"
de_emb_path = "./self-lang-prompt/Llama-2-7b-hf-deu_Latn-32bit"
es_emb_path = "./self-lang-prompt/Llama-2-7b-hf-spa_Latn-32bit"

pathes = [en_emb_path, ar_emb_path, zh_emb_path, jp_emb_path, ru_emb_path, de_emb_path, es_emb_path]

P_1 = []
org_avg = 0
proj_avg = 0

# Set a random seed then calculate
for seed in range(74, 75):
    deta = 0
    cnt = 0
    for i in range(len(pathes)):
        temp = []
        for j in range(len(pathes)):
            src_path = pathes[i]
            tgt_path = pathes[j]
            result = get_acc(src_path, tgt_path, seed, 1000, 'procrustes')
            temp.append("{} / {}".format(result['P@5-org'], result['P@5-proj']))

            if i != j:
                deta += float(result['P@5-proj']) - float(result['P@5-org'])
                proj_avg += float(result['P@5-proj'])
                org_avg += float(result['P@5-org'])
                cnt += 1
        P_1.append(temp)
        print('\t'.join(temp))
    print(deta / cnt)
    print(org_avg / cnt)
    print(proj_avg / cnt)



