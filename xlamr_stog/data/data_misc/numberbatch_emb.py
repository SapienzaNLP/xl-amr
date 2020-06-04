import os


if __name__ == "__main__":

    outdir = "data/numberbatch"
    embeddings_path = os.path.join(outdir,'out_{}.txt')
    with open(os.path.join(outdir, "numberbatch-19.08.en_it_es_de.txt"),"w", encoding="utf-8") as outfile:
        for lang in ["en", "it", "es", "de"]:
            for line in open(embeddings_path.format(lang),"r", encoding="utf-8"):
                fields = line.rstrip().split()
                word = lang+"_"+ fields[0].split("/c/{}/".format(lang))[-1]
                outfile.write("{} {}\n".format(word, " ".join(fields[1:])))
