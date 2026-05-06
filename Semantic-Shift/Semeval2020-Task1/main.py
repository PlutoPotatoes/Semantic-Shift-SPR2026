from extract_embeddings import *
from preprocess_semeval_corpora import *
from calculate_semantic_change import *
from make_semeval_answer_file import *
from get_period_specific_clusters import *
from transformers import AutoModel, AutoTokenizer



#converting each __main__ to a function so we can call them in order from this script's main

def preprocess_semeval_corpora(lang:str = 'english', 
                                corpus_paths:str = 'data/english/english_1.txt;data/english/english_2.txt',
                                target_path:str ='data/english/targets.txt',
                                output_folder:str = 'data/english'):


    target_path = target_path
    corpora = corpus_paths.split(';')
    output_folder = output_folder
    data = []
    outputs = []
    for i, corpus in enumerate(corpora):
        print(lang, output_folder)
        output = open(os.path.join(output_folder, lang + '_preprocessed_' + str(i + 1) + '.txt'), 'a', encoding='utf8')
        outputs.append(output)

    #english only so we have targets
    targets = []
    with open(target_path, 'r', encoding='utf8') as f:
        for line in f:
            target = line.strip()
            if len(target) > 0 :
                targets.append(target)


    for i, corpus in enumerate(corpora):
        with open(corpus, 'r', encoding='utf8') as f:
            for line in f:
                line = filterLine(line.lower(), lang, targets)
                if line is not None:
                    outputs[i].write(line)

    for output in outputs:
        output.close()


def extract_embeddings(batch_size:int=8, 
                       max_length:int = 256, 
                       concat:bool = True, 
                       lang:str = 'english', 
                       gpu:bool = False, 
                       corpus_paths:str = 'data/english/english_preprocessed_1.txt;data/english/english_preprocessed_2.txt',
                       target_path:str ='data/english/targets.txt',
                       model_path:str = '',
                       embeddings_path:str = 'embeddings_english.pickle'):

    #Separate corpus and 
    datasets = corpus_paths.split(';')
    if len(model_path) > 0:
        fine_tuned=True
    else:
        fine_tuned=False

    #we only care about english and pretrained models so were only using that section of the original code

    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModel.from_pretrained(model_path, use_safetensors= True, output_hidden_states=True)
    if gpu:
        model.cuda() # pyright: ignore[reportCallIssue]

    model.eval()

    target_dict = get_targets(target_path, lang)

    get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict=target_dict, concat=concat, gpu=gpu)



def calculate_semantic_change(lang:str = "english",
                              oneEmbPerSentence:bool = True,
                              results_dir:str = "semeval_results/",
                              embeddings_path:str = 'embeddings_english.pickle'):


 
    bert_embeddings = pickle.load(open(embeddings_path, 'rb'))
    target_words = list(bert_embeddings.keys())
    jsd_vec = []
    cosine_dist_vec = []
    results_dict = {"word": [], "aff_prop": [], "kmeans_5":[], "kmeans_7":[], "averaging": [], "aff_prop_clusters":[]}

    sentence_dict = {}
    aff_prop_labels_dict = {}
    aff_prop_centroids_dict = {}
    kmeans_5_labels_dict = {}
    kmeans_5_centroids_dict = {}
    kmeans_7_labels_dict = {}
    kmeans_7_centroids_dict = {}

    aff_prop_pref = -430
    print("Clustering BERT embeddings")
    for i, word in enumerate(target_words):
        print("\n=======", i+1, "- word:", word.upper(), "=======")
        emb = bert_embeddings[word]

        embeddings1 = []
        embeddings2 = []
        texts1 = []
        texts2 = []

        regex = r"\b%s\b" %word.replace("_vb", "").replace("_nn", "")

        print(emb.keys())


        time_slices = ['t1', 't2']
        for ts in time_slices:
            text_seen = {}


            for idx in range(len(emb[ts])):
                ts_text = ts + '_text'
                e = emb[ts][idx]
                text = emb[ts_text][idx]

                if not(re.search(regex, text)):
                    continue

                if oneEmbPerSentence:
                    if text in text_seen:
                        continue
                    else:
                        text_seen[text] = 1

                if ts == 't1':
                    embeddings1.append(e)
                    texts1.append(text)
                elif ts == 't2':
                    embeddings2.append(e)
                    texts2.append(text)


        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)

        print("t1 num. occurences: ", embeddings1.shape[0])
        print("t2 num. occurences: ", embeddings2.shape[0])

        sentence_dict[word] = {time_slices[0]: texts1, time_slices[1]: texts2}

        average_dist = compute_averaged_embedding_dist(embeddings1, embeddings2)

        embeddings_concat = np.concatenate([embeddings1, embeddings2], axis=0)

        aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
        clusters1_aff = list(aff_prop_labels[:embeddings1.shape[0]])
        clusters2_aff = list(aff_prop_labels[embeddings1.shape[0]:])
        n_senses = len(list(set(aff_prop_labels)))
        aff_prop_jsd = compute_divergence_from_cluster_labels(clusters1_aff, clusters2_aff)

        if len(embeddings_concat) >= 5:
            kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=5)
            clusters1_km5 = list(kmeans_5_labels[:embeddings1.shape[0]])
            clusters2_km5 = list(kmeans_5_labels[embeddings1.shape[0]:])
            kmeans5_jsd = compute_divergence_from_cluster_labels(clusters1_km5, clusters2_km5)
        else:
            kmeans_5_labels, kmeans_5_centroids = ['n/a'], ['n/a']
            clusters1_km5 = ['n/a']
            clusters2_km5 = ['n/a']
            kmeans5_jsd = ['n/a']

        #FIXME CHECK TO MAKE SURE YOU HAVE ENOUGH CLUSTERS TO DO THIS, SKIP IF YOU DON'T
        if len(embeddings_concat)>= 7:
            kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, k=7)
            clusters1_km7 = list(kmeans_7_labels[:embeddings1.shape[0]])
            clusters2_km7 = list(kmeans_7_labels[embeddings1.shape[0]:])
            kmeans7_jsd = compute_divergence_from_cluster_labels(clusters1_km7, clusters2_km7)
        else:
            kmeans_7_labels, kmeans_7_centroids = ['n/a'], ['n/a']
            clusters1_km7 = ['n/a']
            clusters2_km7 = ['n/a']
            kmeans7_jsd = ['n/a']

        # add results to dataframe for saving
        aff_prop_labels_dict[word] = {time_slices[0]: clusters1_aff, time_slices[1]: clusters2_aff}
        aff_prop_centroids_dict[word] = aff_prop_centroids

        kmeans_5_labels_dict[word] = {time_slices[0]: clusters1_km5, time_slices[1]: clusters2_km5}
        kmeans_5_centroids_dict[word] = kmeans_5_centroids

        kmeans_7_labels_dict[word] = {time_slices[0]: clusters1_km7, time_slices[1]: clusters2_km7}
        kmeans_7_centroids_dict[word] = kmeans_7_centroids  # add results to dataframe for saving

        results_dict["word"].append(word)
        results_dict["aff_prop"].append(aff_prop_jsd)
        results_dict["aff_prop_clusters"].append(n_senses)
        results_dict["kmeans_5"].append(kmeans5_jsd)
        results_dict["kmeans_7"].append(kmeans7_jsd)
        results_dict["averaging"].append(average_dist)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        csv_file = results_dir + "results_" + lang + ".csv"
        labels_file = results_dir + "labels_" + lang + ".pkl"
        centroids_file = results_dir + "centroids_" + lang + ".pkl"
        sents_file = results_dir + "sents_" + lang + ".pkl"

        # save results to CSV
        results_df = pd.DataFrame.from_dict(results_dict)
        results_df = results_df.sort_values(by=['aff_prop'], ascending=False)
        results_df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)

        # save cluster labels to pickle
        labels_file = results_dir + "aff_prop_labels_" + lang + ".pkl"
        centroids_file = results_dir + "aff_prop_centroids_" + lang + ".pkl"
        pf = open(labels_file, 'wb')
        pickle.dump(aff_prop_labels_dict, pf)
        pf.close()
        pf2 = open(centroids_file, 'wb')
        pickle.dump(aff_prop_centroids_dict, pf2)
        pf2.close()

        labels_file = results_dir + "kmeans_5_labels_" + lang + ".pkl"
        centroids_file = results_dir + "kmeans_5_centroids_" + lang + ".pkl"
        pf = open(labels_file, 'wb')
        pickle.dump(kmeans_5_labels_dict, pf)
        pf.close()
        pf2 = open(centroids_file, 'wb')
        pickle.dump(kmeans_5_centroids_dict, pf2)
        pf2.close()

        labels_file = results_dir + "kmeans_7_labels_" + lang + ".pkl"
        centroids_file = results_dir + "kmeans_7_centroids_" + lang + ".pkl"
        pf = open(labels_file, 'wb')
        pickle.dump(kmeans_7_labels_dict, pf)
        pf.close()
        pf2 = open(centroids_file, 'wb')
        pickle.dump(kmeans_7_centroids_dict, pf2)
        pf2.close()

        # save sentences
        pf3 = open(sents_file, 'wb')
        pickle.dump(sentence_dict, pf3)
        pf3.close()
        print("Done! Saved results in", csv_file, "!")

def make_semeval_answer_file(lang:str = "english",
                            cluster_method:str = 'aff_prop',
                            results_dir:str = "semeval_results/results_english.csv",
                            target_path:str = 'data/english/targets.txt'):

    english_threshold = 0.3
    method_name = cluster_method
    methods = ['aff_prop', 'kmeans_5', 'kmeans_7', 'averaging']
    if method_name not in methods:
        print("Method not valid, valid choices are: ", ", ".join(methods))
        sys.exit()

    print("Language:", lang.upper())

    clustering_file = results_dir
    clustering_df = pd.read_csv(clustering_file, sep="\t")
    target_file = target_path
    target_words = open(target_file,'r').readlines()
    target_words = [t.strip() for t in target_words]

    thresh = english_threshold
    if not os.path.exists("answer/task1/"):
        os.makedirs("answer/task1/")
    outfilename1 = "answer/task1/" + lang  + ".txt"
    outfile1 = open(outfilename1, 'w', encoding='utf-8')
    for i,word in enumerate(target_words):
        val = float(clustering_df[clustering_df['word'] == word][method_name])
        classif = 0 if val < thresh else 1
        line = word + "\t" + str(classif)
        outfile1.write(line)
        if i < len(target_words):
            outfile1.write("\n")
    print("Done writing", outfilename1,"!")

    if not os.path.exists("answer/task2/"):
        os.makedirs("answer/task2/")
    outfilename2 = "answer/task2/" + lang  + ".txt"
    outfile2 = open(outfilename2, 'w', encoding='utf-8')
    for i,word in enumerate(target_words):
        val = float(clustering_df[clustering_df['word'] == word][method_name])
        line = word + "\t" + str(val)
        outfile2.write(line)
        if i < len(target_words):
            outfile2.write("\n")

    print("Done writing", outfilename2,"!")


def get_period_specific_clusters(lang:str = 'english',
                                 target_path:str = 'data/english/targets.txt',
                                 results_path:str = "semeval_results/kmeans_7_labels_english.pkl",
                                 dynamic_treshold:bool = True,
                                 treshold:int = 2):


    target_dict = get_targets(target_path, lang)
    targets = target_dict.values()

    changed = classify(targets, lang, results_path, dynamic_treshold, treshold)
    write_to_file(lang, changed, target_path)


if __name__ == '__main__':
    #Only need to run this once to create the preprocessed file. This filters out lines that don't contain our target tokens
    #before running make sure you create data/english/ and include targets.txt + ccoha1.txt and ccoha2.txt as english_1.txt and english_2.txt
    
    preprocess_semeval_corpora(lang = 'english', 
                                corpus_paths = 'data/english/english_1.txt;data/english/english_2.txt',
                                target_path ='data/english/targets.txt',
                                output_folder = 'data/english/')
    print('corpora processed')    
    
    extract_embeddings(batch_size=8, 
                       max_length = 256, 
                       concat = True, 
                       lang = 'english', 
                       gpu = False, 
                       corpus_paths = 'data/english/english_preprocessed_1.txt;data/english/english_preprocessed_2.txt',
                       target_path ='data/english/targets.txt',
                       model_path = 'model/',
                       embeddings_path = 'embeddings_english.pickle')
    print('embeddings extracted')
    
    calculate_semantic_change(lang = "english",
                              oneEmbPerSentence = True,
                              results_dir = "semeval_results/",
                              embeddings_path = 'embeddings_english.pickle')
    print('change calculated')

    make_semeval_answer_file(lang = "english",
                            cluster_method = 'aff_prop',
                            results_dir = "semeval_results/results_english.csv",
                            target_path = 'data/english/targets.txt')
    print('answers recorded')

    get_period_specific_clusters(lang = 'english',
                                 target_path = 'data/english/targets.txt',
                                 results_path = "semeval_results/kmeans_7_labels_english.pkl",
                                 dynamic_treshold = True,
                                 treshold = 2)