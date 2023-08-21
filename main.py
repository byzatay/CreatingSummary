# Arayüz için kütüphaneler
import re

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys

# Graf için kütüphaneler
import networkx as nx
import matplotlib.pyplot as plt

# Cümle düzenleme işlemleri için kütüphaneler
import nltk
import string
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

# Benzerlik için kütüphaneler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Rouge için kütüphaneler
from sumeval.metrics.rouge import RougeCalculator

from math import log10

from arayuz import Ui_MainWindow


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.document = None
        self.real_summary = None
        self.title = None
        self.our_summary = None
        self.original_sentences = []
        self.clean_sentences = []
        self.sentence_similarity_matrix = None
        self.sentence_score_array = []
        self.similarity_treshold = None
        self.score_treshold = None
        # Benzerlik oranını geçen cümlelerin node bağlantı sayıları için
        self.passing_node_threshold_num = []

        self.ui.dcmtgrs.clicked.connect(self.read_document)
        self.ui.createozet.clicked.connect(self.find_sentence_similarty)
        self.ui.ozetgrs.clicked.connect(self.read_summary)

    # Doküman seçimi ve dokümanın içindeki verinin okunması
    def read_document(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Dosya Seç", "", "Tüm Dosyalar (*)")
        with open(file_path, 'r') as f:
            self.document = f.read()
            self.ui.textEdit.setPlainText(self.document)

        self.clean_document()

    # Gerçek özetin seçilip okunması
    def read_summary(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Dosya Seç", "", "Tüm Dosyalar (*)")
        with open(file_path, 'r') as f:
            self.real_summary = f.read()
            self.ui.textEdit_3.setPlainText(self.real_summary)
        self.find_rouge(self.real_summary, self.our_summary)

    # Doküman içeriğinin düzenlenmesi ve başlığın bulunması
    def clean_document(self):
        lines = self.document.split('\n')
        self.title = lines[0]

        text_without_title = lines[1:]
        text_without_empty_lines = [item for item in text_without_title if item != ""]

        for sentence in text_without_empty_lines:
            sentence = sentence.split(".")
            for cumle in sentence:
                if cumle:
                    cumle = cumle.lstrip()
                    self.clean_sentences.append(cumle)

        self.original_sentences = self.clean_sentences.copy()

    # Dokümanın grafının çizilmesi
    def drawgraph(self):
        # Grafta orijinal cümlelerin gözükmesi için
        # sentences = [sentences.strip() for sentences in self.original_sentences]

        sentences = []
        for i in range(len(self.original_sentences)):
            temp_str = "Cümle "+str(i+1)
            sentences.append(temp_str)

        G = nx.Graph()

        # Cümleleri düğüm olarak ekle
        # Düğümlere skor ve benzerlik oranını geçen düğüm bilgisini ata
        for sentence in sentences:
            index = sentences.index(sentence)
            score = "{:.4f}".format(self.sentence_score_array[index])
            node_num = self.passing_node_threshold_num[index]
            G.add_node(sentence, weight=score, value=node_num)

        # Düğümler arasındaki kenarları ekle
        # Kenarlara benzerlik bilgisini ekle
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # if float(similarity[i][j]) > 0.4:
                benzerlik = "{:.4f}".format(self.sentence_similarity_matrix[i][j])
                G.add_edge(sentences[i], sentences[j], weight=benzerlik)

        node_colors = []
        edge_colors = []

        # Skor değerlerinin tresholdu geçip geçmediği kontrolü
        for node in G.nodes:
            weight = G.nodes[node]['weight']
            # value = G.nodes[node]['value']
            if float(weight) > self.score_treshold:
                node_colors.append('green')
            else:
                node_colors.append('lightblue')

        # Benzerlik değerlerinin tresholdu geçip geçmediği kontrolü
        for edge in G.edges:
            weight = G.edges[edge]['weight']
            if float(weight) > self.similarity_treshold:
                edge_colors.append('red')
            else:
                edge_colors.append('gray')

        # Düğümlerin üzerinde özellikleri gösterme
        node_labels = {node: f"{node}\nSkor: {G.nodes[node]['weight']}\nNode: {G.nodes[node]['value']}" for node in
                       G.nodes}

        # Kenarların üzerinde özellikleri gösterme
        edge_labels = {(u, v): G.edges[u, v]['weight'] for u, v in G.edges}

        # Grafın çizilmesi
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=500,
                         font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors)

        plt.axis('off')
        plt.title('Doküman Grafı')
        plt.show()

    # Cümle benzerliğinin bulunması için cümlelerin temizlenmesi
    def find_sentence_similarty(self):
        self.similarity_treshold = float(self.ui.benzerlik.text())
        self.score_treshold = float(self.ui.skor.text())

        # Nltk kütüphanelerinin yüklenmesi
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')
        # nltk.download('stopwords')

        nltk_sentences = []
        for sentence in self.clean_sentences:
            # Tokenize işlemi (Stringin kelimelere ayrılması)
            sentence = sentence.lower()
            tokens = nltk.word_tokenize(sentence)

            # Punctuation işlemi (Noktalama işaretlerinin silinmesi)
            punctuations = list(string.punctuation)
            punctuations.append("``")
            punctuations.append("''")
            filtered_words_no_punct = []

            for token in tokens:
                if token not in punctuations:
                    filtered_words_no_punct.append(token)

            # Stop-wordlerin kaldırılması işlemi
            stop_words = set(stopwords.words('english'))
            stop_words.add("'s")
            stop_words.add("'t")
            filtered_tokens = [filtered_words_no_punct for filtered_words_no_punct in filtered_words_no_punct if
                               not filtered_words_no_punct in stop_words]

            # Stemming işlemi (Kelimelerin köklerinin bulunması)
            root_tokens = []
            for token in filtered_tokens:
                lemmatizer = WordNetLemmatizer()
                root_word = lemmatizer.lemmatize(token)
                root_tokens.append(root_word)

            clear_sentence = " ".join(root_tokens)
            nltk_sentences.append(clear_sentence)

        self.clean_sentences = nltk_sentences.copy()

        self.find_similarity_with_bert()
        self.find_sentence_score()
        self.find_summary()
        self.drawgraph()

    # Cümle benzerliğinin bulunması
    def find_similarity_with_bert(self):
        rows = len(self.clean_sentences)
        cols = rows
        self.sentence_similarity_matrix = [[0] * cols for _ in range(rows)]

        model = SentenceTransformer('nli-distilroberta-base-v2')
        sentence_embeddings = model.encode(self.clean_sentences)

        for i in range(0, len(self.clean_sentences)):
            similarity_score = cosine_similarity(
                [sentence_embeddings[i]],
                sentence_embeddings[0:]
            ).flatten()
            self.sentence_similarity_matrix[i] = similarity_score.copy()

        """rows = len(self.sentence_similarity_matrix)
        cols = len(self.sentence_similarity_matrix[0])

        for i in range(rows):
            for j in range(cols):
                print(self.sentence_similarity_matrix[i][j], end=" ")
            print()"""

    # Cümle skorunun bulunması
    def find_sentence_score(self):

        # P1 = Cümledeki özel isim sayısı / Cümlenin uzunluğu
        # P2 = Cümledeki numerik veri sayısı / Cümlenin uzunluğu
        # P3 = Cümle benzerliği tresholdunu geçen nodeların bağlantı sayısı / Toplam bağlantı sayısı
        # P4 = Cümledeki başlıkta geçen kelime sayısı / Cümlenin uzunluğu
        # P5 = Cümlenin içinde geçen tema kelime sayısı / Cümlenin uzunluğu

        for sentence in self.original_sentences:
            p1 = self.find_p1(sentence)
            p2 = self.find_p2(sentence)
            index = self.original_sentences.index(sentence)
            p3 = self.find_p3(index)
            index = self.original_sentences.index(sentence)
            clean_sentence = self.clean_sentences[index]
            p4 = self.find_p4(clean_sentence, sentence)
            p5 = self.find_p5(clean_sentence, sentence)
            score = (p1+p2+p3+p4+p5)/5
            self.sentence_score_array.append(score)

    def find_p1(self, sentence):
        # nltk.download('averaged_perceptron_tagger')

        ozelisim = 0

        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)

        for word, tag in tagged:
            if tag == 'NNP':
                ozelisim += 1

        p1 = ozelisim / len(words)

        return p1

    def find_p2(self, sentence):
        numbers = re.findall(r'\d+', sentence)
        count_nums = len(numbers)
        p2 = count_nums / len(sentence.split())
        return p2

    def find_p3(self, index):
        count_treshold = 0
        for i in range(0, len(self.original_sentences)):
            if i != index:
                if self.sentence_similarity_matrix[index][i] > self.similarity_treshold:
                    count_treshold += 1

        self.passing_node_threshold_num.append(count_treshold)
        p3 = count_treshold / len(self.original_sentences)
        return p3

    def find_p4(self, clean_sentence, sentence):
        title_words = self.title.lower().split()
        title_root_tokens = []
        for token in title_words:
            lemmatizer = WordNetLemmatizer()
            root_word = lemmatizer.lemmatize(token)
            title_root_tokens.append(root_word)

        sentence_words = clean_sentence.lower().split()

        matching_words = []

        for word in title_words:
            if word in sentence_words:
                matching_words.append(word)

        length = len(sentence.split())
        p4 = len(matching_words) / length
        return p4

    def find_p5(self, clean_sentence, sentence):
        tema_words = []
        tema_words = self.find_tf_idf()

        tema_words_num = 0
        for word in tema_words:
            if word in clean_sentence:
                tema_words_num += clean_sentence.count(word)

        length = len(sentence.split())
        p5 = tema_words_num / length
        return p5

    # TF-IDF değerlerinin ve tema kelimelerinin bulunması
    def find_tf_idf(self):
        unique_words = []

        for sentence in self.clean_sentences:
            words = sentence.split(" ")

            # Eşsiz kelimelerin bulunması
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)

        cols = len(unique_words)
        rows = len(self.clean_sentences)
        tfidf_matrix = [[0] * cols for _ in range(rows)]

        # Kelimelerin frekanslarının bulunması
        # Bu aşamanın sonunda tfidf_matrix içinde frekans bilgisi tutulur
        for sentence in self.clean_sentences:
            x = self.clean_sentences.index(sentence)
            sentence = sentence.split(" ")

            for word in unique_words:
                y = unique_words.index(word)
                frekans = sentence.count(word)
                tfidf_matrix[x][y] = frekans

        # TF değerinin hesaplanması
        # Bu aşamanın sonunda tfidf_matrix içinde tf değeri bilgisi tutulur
        for i in range(rows):
            length = len(self.clean_sentences[i].split(" "))
            for j in range(cols):
                tfidf_matrix[i][j] /= length

        # Kelimelerin kaç dokümanda geçtiğinin bulunması
        # Bu aşamanın sonunda idf dizisinde kelimenin kaç dokümanda geçtiği bilgisi tutulur
        idf = []
        count = 0
        for key in unique_words:
            for sentence in self.clean_sentences:
                sentence = sentence.split(" ")
                if key in sentence:
                    count += 1

            idf.append(count)
            count = 0

        # DF değerinin bulunması
        df = []
        for i in range(len(idf)):
            df.append(len(self.clean_sentences) / idf[i])

        # IDF değerinin bulunması
        for i in range(len(idf)):
            idf[i] = log10(df[i])

        # TF-IDF değerinin hesaplanması
        tfidf_matrix = [[float(value) for value in row] for row in tfidf_matrix]
        for i in range(len(tfidf_matrix)):
            for j in range(len(tfidf_matrix[i])):
                tfidf_matrix[i][j] = tfidf_matrix[i][j] * idf[j]

        """for i in range(rows):
            for j in range(cols):
                print(tfidf_matrix[i][j], end=" ")
            print()"""

        # Kelimelerin ortalama TF-IDF değerinin hesaplanması
        average_tfidf = [0] * len(tfidf_matrix[0])

        for i in range(rows):
            for j in range(cols):
                average_tfidf[j] += tfidf_matrix[i][j]

        for i in range(len(average_tfidf)):
            average_tfidf[i] /= rows

        # Tema kelimelerin bulunması
        tema_words = []
        tema_words_tfidf_degerleri = []

        for i in range(rows):
            for j in range(cols):
                if tfidf_matrix[i][j] > average_tfidf[j]:
                    if unique_words[j] not in tema_words:
                        tema_words.append(unique_words[j])
                        tema_words_tfidf_degerleri.append(average_tfidf[j])

        sort = sorted(range(len(tema_words_tfidf_degerleri)), key=lambda k: tema_words_tfidf_degerleri[k], reverse=True)
        sorted_tema_words_ifidf = sorted(tema_words_tfidf_degerleri, reverse=True)
        sorted_tema_words = [tema_words[i] for i in sort]

        tema_words_num = 0

        for sentence in self.clean_sentences:
            words = sentence.split()
            tema_words_num += len(words)

        tema_words_num = round(tema_words_num*0.1)
        tema_words = sorted_tema_words[:tema_words_num]
        return tema_words

    def find_summary(self):
        node_scores = [[0] * 2 for _ in range(len(self.original_sentences))]
        for i in range(len(self.original_sentences)):
            node_scores[i][0] = self.passing_node_threshold_num[i]
            node_scores[i][1] = i+1

        summary = self.original_sentences.copy()

        sort = sorted(zip(node_scores, summary), key=lambda x: x[0][0], reverse=True)
        sorted_node_score, sorted_summary = zip(*sort)

        summary.clear()
        summary = [str(eleman) for eleman in sorted_summary]
        node_scores.clear()
        node_scores = list(sorted_node_score)

        sum_sen_num = round(len(self.original_sentences)*0.40)

        passing_treshold_index = []
        new_summary = []

        for i in range(len(self.sentence_score_array)):
            if self.sentence_score_array[i] > self.score_treshold:
                passing_treshold_index.append(i+1)

        if len(passing_treshold_index) > sum_sen_num:
            index = 0
            temp_array = node_scores.copy()
            for i in range(len(node_scores)):
                if node_scores[i][1] not in passing_treshold_index:
                    aranan = node_scores[i][1]
                    for j, alt_liste in enumerate(temp_array):
                        if alt_liste[1] == aranan:
                            index = j
                            del temp_array[j]
                            break
                    summary.pop(index)

            summary = summary[:sum_sen_num]
            new_summary = '. '.join(summary)
            self.our_summary = new_summary

        else:
            index = 0
            for i in range(len(passing_treshold_index)):
                aranan = passing_treshold_index[i]
                for j, alt_liste in enumerate(node_scores):
                    if alt_liste[1] == aranan:
                        index = j
                        break
                new_summary.append(summary[index])

            kalan = sum_sen_num - len(passing_treshold_index)

            for j in range(len(node_scores)):
                if node_scores[j][1] not in passing_treshold_index:
                    if kalan == 0:
                        break
                    else:
                        aranan = node_scores[j][1]
                        for i, alt_liste in enumerate(node_scores):
                            if alt_liste[1] == aranan:
                                index = i
                                break

                        new_summary.append(summary[index])
                        kalan -= 1

            new_summary = '. '.join(new_summary)
            self.our_summary = new_summary

        self.ui.textEdit_2.setPlainText(self.our_summary)

    def find_rouge(self, real_summary, summary):
        rouge = RougeCalculator()
        rouge_skoru = rouge.rouge_n(
            summary=summary,
            references=real_summary,
            n=1
        )
        self.ui.textEdit_4.setPlainText(str(rouge_skoru))


# Uygulamanın çalıştırılması
def app():
    app1 = QtWidgets.QApplication(sys.argv)
    win = MyApp()
    win.show()
    sys.exit(app1.exec())


app()
