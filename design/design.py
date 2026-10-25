import sys
from pathlib import Path

# Project root first, so `utils`/`models` resolve there; `mcts`/`seq_utils` resolve here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import torch
import warnings, os
import pandas as pd
from experiment import load_config
from models import PRIDA
from pre_train.train import ESMEncoder, RNAEncoder, ProjectionHead, masked_mean_pool

from time import time

from mcts import MCTS
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

def evaluate_candidate(rna_encoder, rna_proj, classifier, mcts, target_feature, verbose):
    classifier.eval()
    with torch.no_grad():
        sim_seq = mcts.get_candidate()
        apte_tokens, apte_mask = rna_encoder([sim_seq])
        apte_feature = masked_mean_pool(rna_proj(apte_tokens), apte_mask).to(device)

        target_feature = target_feature.to(device)
        score = classifier.predict_score(apte_feature,target_feature)


    if verbose:
        print(f"Candidate: {mcts.get_candidate()}\tScore: {score}")
        print("*" * 80)

    mcts.reset()

    return score

def recommend(prot_encoder, prot_proj, rna_encoder, rna_proj, classifier, device, target, n_aptamers, top_k, depth, iteration, verbose=True):

    classifier = classifier.to(device)
    
    results = {}
    
    with torch.no_grad():
        prot_tokens, prot_mask = prot_encoder([target])
        target_feature = masked_mean_pool(prot_proj(prot_tokens), prot_mask)
    
    

        mcts = MCTS(target_feature, depth=depth, iteration=iteration, 
                    states=8, target_protein=target, device=device)
        all_candidates = []
        

        for i in range(n_aptamers):
            mcts.make_candidate(classifier, rna_encoder, rna_proj)
            candidate = mcts.get_candidate()
            score = evaluate_candidate(rna_encoder, rna_proj, classifier, mcts, target_feature, verbose)
            candidate_data = {'candidate': candidate, 'score': score}
            all_candidates.append(candidate_data)
            
    sorted_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)

    top_candidates = sorted_candidates[:top_k]

    for i, candidate_data in enumerate(top_candidates):
        results[i] = candidate_data
        

    return results

def main():

    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    proj_dir = "/path/to/your/PRIDA/pre_train/dataset_name/best_pretrain_proj.pt"
    ckpt = torch.load(proj_dir, map_location=device)

    rna_model_dir = "/path/to/your/PRIDA/checkpoints/RiNALMo"
    rna_encoder = RNAEncoder(rna_model_dir).to(device)
    rna_proj = ProjectionHead(
        input_dim=rna_encoder.output_dim,
        output_dim=512
    ).to(device)
    rna_proj.load_state_dict(ckpt["rna_proj"])
    rna_proj.eval()

    prot_model_dir = "/path/to/your/PRIDA/checkpoints/ESM2"
    prot_encoder = ESMEncoder(prot_model_dir).to(device)
    prot_encoder.eval()
    prot_proj = ProjectionHead(
        input_dim=prot_encoder.output_dim,
        output_dim=512
    ).to(device)    
    prot_proj.load_state_dict(ckpt["prot_proj"], strict=True)
    prot_proj.eval()


    # Written by `python main.py --data dataset_name --fold k --save-model`.
    model_path = "/path/to/your/PRIDA/runs/final/dataset_name/fold_1/seed_42/best_model.pt"
    config_path = "/path/to/your/PRIDA/configs/dataset_name.yaml"
    checkpoint = torch.load(model_path, map_location=device)
    model = PRIDA(**load_config(config_path)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()


    target_protein = "FNLDVDSPAEYSGPEGSYFGFAVDFFVPSASSRMFLLVGAPKANTTQPGIVEGGQVLKCDWSSTRRCQPIEFDATGNRDYAKDDPLEFKSHQWFGASVRSKQDKILACAPLYHWRTEMKQEREPVGTCFLQDGTKTVEYAPCRSQDIDADGQGFCQGGFSIDFTKADRVLLGGPGSFYWQGQLISDQVAEIVSKYDPNVYSIKYNNQLATRTAQAIFDDSYLGYSVAVGDFNGDGIDDFVSGVPRAARTLGMVYIYDGKNMSSLYNFTGEQMAAYFGFSVAATDINGDDYADVFIGAPLFMDRGSDGKLQEVGQVSVSLQRASGDFQTTKLNGFEVFARFGSAIAPLGDLDQDGFNDIAIAAPYGGEDKKGIVYIFNGRSTGLNAVPSQILEGQWAARSMPPSFGYSMKGATDIDKNGYPDLIVGAFGVDRAILYRARPVITVNAGLEVYPSILNQDNKTCSLPGTALKVSCFNVRFCLKADGKGVLPRKLNFQVELLLDKLKQKGAIRRALFLYSRSPSHSKNMTISRGGLMQCEELIAYLRDESEFRDKLTPITIFMEYRLDYRTAADTTGLQPILNQFTPANISRQAHILLDCGEDNVCKPKLEVSVDSDQKKIYIGDDNPLTLIVKAQNQGEGAYEAELIVSIPLQADFIGVVRNNEALARLSCAFKTENQTRQVVCDLGNPMKAGTQLLAGLRFSVHQQSEMDTSVKFDLQIQSSNLFDKVSPVVSHKVDLAVLAAVEIRGVSSPDHVFLPIPNWEHKENPETEEDVGPVVQHIYELRNNGPSSFSKAMLHLQWPYKYNNNTLLYILHYDIDGPMNCTSDMEINPLRIKISSLQTTEKNDTVAGQGERDHLITKRDLALSEGDIHTLGCGVAQCLKIVCQVGRLDRGKSAILYVKSLLWTETFMNKENQNHSYSLKSSASFNVIEFPYKNLPIEDITNSTLVTTNVTWGIQP"

    save_path = "aptamer.txt"

    recommended_aptamers = recommend(
        prot_encoder=prot_encoder,
        prot_proj=prot_proj,
        rna_encoder=rna_encoder,
        rna_proj=rna_proj,
        classifier=model,
        device=device,
        target=target_protein,
        n_aptamers=15,
        top_k=10,
        depth=20,
        iteration=200,
        verbose=True
    )

    with open(save_path, "w", encoding="utf-8") as f:
        sorted_aptamers = sorted(
            recommended_aptamers.values(),
            key=lambda x: x["score"],  
            reverse=True
        )
        
        for aptamer_data in sorted_aptamers:
            f.write(f"{aptamer_data['candidate']}\n")


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")