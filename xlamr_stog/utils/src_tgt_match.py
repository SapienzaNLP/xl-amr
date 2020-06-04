import re
from scipy import spatial
from collections import Counter

POS_BN ={"PROPN":"NOUN","PRON":"NOUN"}
def compute_cos_sim(emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1
    result = 1 - spatial.distance.cosine(emb1, emb2)
    return result

def closest_in_emb_space(t, tgt_token, language, lang_en_mapping):
    # lang_en_mapping keys have the lang_ prefix
    match = False
    lang_token =language+"_"+t
    tk = ""
    if lang_token in lang_en_mapping:
        tk = lang_token
    elif lang_token.lower() in lang_en_mapping:
        tk = lang_token.lower()
    if tk != "":
        if len(lang_en_mapping[tk]) > 0:
            for neighbor in range(len(lang_en_mapping[tk])):
                en_token, sim = lang_en_mapping[tk][neighbor]
                if sim >=0.60:
                    tr_token = en_token[3:]

                    if tgt_token == tr_token:
                        match = True
                        break
    return match

def closet_in_src(src_tokens,tgt_token, src_pos, language, translation_emb, lexicalizations, missing_lex):
    en_tgt = "en_"+tgt_token
    src_tokens_lang = [language+"_"+t for t in src_tokens]
    if en_tgt not in translation_emb:
        return None
    tgt_emb = translation_emb[en_tgt]
    src_emb = [translation_emb[t] if t in translation_emb else None for t in src_tokens_lang]
    sim_tgt_src = [compute_cos_sim(tgt_emb, s_emb) for s_emb in src_emb]
    sorted_src_tokens_sim, sorted_src_tokens = zip(*sorted(zip(sim_tgt_src, src_tokens), reverse=True))
    if lexicalizations is not None:
        for i, token in enumerate(sorted_src_tokens[:5]):
            postag = POS_BN[src_pos[token]] if src_pos[token] in POS_BN else src_pos[token]
            lemma_pos = token+"#"+postag.lower()
            if lemma_pos in lexicalizations:
                tok_lex = lexicalizations[lemma_pos]
                tok_lex_words ={word.split("#")[0] for word in tok_lex}
                if tgt_token in tok_lex_words:
                    return token
            else:
                if lemma_pos not in missing_lex:
                    missing_lex[lemma_pos]=0
                missing_lex[lemma_pos]+=1
    # else:
    if sorted_src_tokens_sim[0]>=0.60:
        return sorted_src_tokens[0]

def find_tgt_token_src(tgt_token, src_tokens, src_postags=None, language=None, translation_mappings=None, replacements=None, training=False):
    tgt_token = re.sub(r'-\d\d$', '', tgt_token)  # .lower())
    if tgt_token.startswith('en_'):
        tgt_token=tgt_token[3:]
    if tgt_token in ["@start@","@end@"]: return None
    return_token = None
    for i, token in enumerate(src_tokens):
        if tgt_token == token:
            return_token = token
            break
        # elif tgt_token.startswith('en_'):
        #     if tgt_token[3:]==token:
        #         return_token = token
        #         break
        closest_by_emb = closest_in_emb_space(token, tgt_token, language, translation_mappings[0][language])
        if closest_by_emb:
            return_token=token
            break
    if return_token is not None:
        if training:
            if return_token not in replacements:
                replacements[return_token] = Counter()
            replacements[return_token].update([tgt_token])
        return return_token

    if translation_mappings[1] is not None:
        closest_in_src = closet_in_src(src_tokens,tgt_token, src_postags, language, translation_mappings[1], translation_mappings[2][language], translation_mappings[3][language])
        if closest_in_src is not None:
            if training:
                if closest_in_src not in replacements:
                    replacements[closest_in_src] = Counter()
                replacements[closest_in_src].update([tgt_token])
            return closest_in_src

    return None


def find_similar_token_vector_fixed(lang_token, lang_en_mapping):
    tk = ""
    if lang_token in lang_en_mapping:
        tk = lang_token
    elif lang_token.lower() in lang_en_mapping:
        tk = lang_token.lower()
    if tk!="":
        if len(lang_en_mapping[tk])>0:
            en_token, sim = lang_en_mapping[tk][0]
            if sim >=0.60:
                return en_token[3:]
    return None

def find_similar_token_vector_test(lang_token, precomputed_mappings):
    lang_en_replacement_rules, lang_en_mapping = precomputed_mappings
    tk = ""
    if lang_token[3:] in lang_en_replacement_rules:
        tk = lang_token[3:]
    elif lang_token[3:].lower() in lang_en_replacement_rules:
        tk = lang_token[3:].lower()
    if tk!="":
        if len(lang_en_replacement_rules[tk])>0:
            en_token = lang_en_replacement_rules[tk].most_common()[0][0]
            return en_token
    return find_similar_token_vector_fixed(lang_token,lang_en_mapping)

