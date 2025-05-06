import jieba
import jieba.posseg as psg
from gensim import corpora, models
from data_load import load_pdf
import os
import pickle
from scipy.special import softmax


# 停用词表加载方法
def get_stopword_list(stop_word_path="public/stopwords/stopwords_baidu.txt"):
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
# jieba词性标注表详见项目根目录doc/词性标注表.md
def word_filter(_seg_list, pos_type: str = None):
    stopword_list = get_stopword_list()
    res = []
    for seg in _seg_list:
        # 根据pos_type参数选择是否词性过滤
        # 不进行词性过滤，表示全部保留
        if pos_type is None:
            word = seg
            flag = None
        else:
            word = seg.word
            flag = seg.flag

        if flag is not None and not flag.startswith(pos_type):
            continue
        else:
            # 过滤停用词表中的词，以及长度为<2的词
            if not word in stopword_list and len(word) > 1:
                res.append(word)

    return res


def find_files(directory, fe: str):
    res_files = []
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为 .pdf
            if file.lower().endswith(fe):
                # 构造完整的文件路径
                full_path = os.path.join(root, file)
                res_files.append(full_path)
    return res_files


class Object:
    def __init__(self):
        self.path = ""

    def save_self(self, save_path=None):
        if save_path is None:
            with open(self.path, "wb") as f:
                pickle.dump(self, f)
        else:
            # 保存对象到文件
            with open(save_path, "wb") as f:
                pickle.dump(self, f)

    def load_self(self, load_path=None):
        if load_path is None:
            with open(self.path, "rb") as f:
                return pickle.load(f)
        else:
            # 从文件中加载对象
            with open(load_path, "rb") as f:
                return pickle.load(f)


class Corpus(Object):
    def __init__(self):
        super().__init__()
        # 原始的语料
        self.raw_documents = []
        # 经初步处理后（分词、停用词及词性过滤）的语料
        self.documents = []
        self.path = "object/corpus.pickle"

    def load_from_pdf(self, directory: str = "public/raw_corpus", _pos=False):
        # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
        for pdf in find_files(directory, ".pdf"):
            pdf_string = load_pdf(pdf)
            content = pdf_string
            self.raw_documents.append(content)
            seg_list = seg_to_list(content, pos=True)
            filter_list = word_filter(seg_list, pos_type="n")
            self.documents.append(filter_list)

    # def load_from_csv(self, directory: str = "../public/raw_corpus", _pos=False):
    #     # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    #     pdf = 1
    #     pdf_string = load_pdf(pdf)
    #     content = pdf_string
    #     self.raw_documents.append(content)
    #     seg_list = seg_to_list(content, pos=True)
    #     filter_list = word_filter(seg_list, pos_type="n")
    #     self.documents.append(filter_list)


# 语料特征（word）的索引字典
class DictCorpus(Corpus):
    def __init__(self):
        super().__init__()
        # 词空间
        self.dictionary = None
        self.path = "object/dict_corpus.pickle"
        self.corpus = None
        self.tfidf = None

    # 将文档转为词空间
    def to_dict(self):
        res = corpora.Dictionary(self.documents)
        self.dictionary = res

    # 使用BOW模型向量化
    def document_to_vector(self):
        corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        self.corpus = corpus
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        self.tfidf = corpus_tfidf
        return corpus_tfidf


# 主题模型（LSI、LDA）
class TopicModel:
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, dict_corpus: DictCorpus, model='LSI', num_topics=20):
        self.dc = dict_corpus
        self.path = "object/model.model"
        self.word_topic_dict = {}
        self.num_topics = num_topics
        self.model_name = model
        self.model = None

    def train(self):
        # 选择模型
        if self.model_name == 'LSI':
            self.model = models.LsiModel(self.dc.tfidf, id2word=self.dc.dictionary, num_topics=self.num_topics)
        else:
            self.model = models.LdaModel(self.dc.tfidf, id2word=self.dc.dictionary, num_topics=self.num_topics)
        self.model.save(self.path)

    def load(self):
        if self.model_name == 'LSI':
            self.model = models.LsiModel(self.dc.tfidf, id2word=self.dc.dictionary).load(self.path)
        else:
            self.model = models.LdaModel(self.dc.tfidf, id2word=self.dc.dictionary).load(self.path)

    def predict_topic(self, doc: str):
        """
        预测新文档对主题的分布概率

        参数:
            doc: 文档

        返回:
            [(主题ID, 概率)]
        """
        seg_list = seg_to_list(doc, pos=True)
        filter_list = word_filter(seg_list, pos_type="n")
        doc_bow = self.dc.dictionary.doc2bow(filter_list)

        if self.model_name == 'LSI':
            # LSI模型可以直接使用原始TF-IDF向量
            tfidf_model = models.TfidfModel(self.dc.corpus)
            doc_tfidf = tfidf_model[doc_bow]
            topic_dist = self.model[doc_tfidf]
        else:
            # LDA模型直接使用词袋向量
            topic_dist = self.model[doc_bow]

        # 对LDA结果已经是概率分布，对LSI需要softmax归一化
        if self.model_name == 'LSI':
            topic_dist = sorted(topic_dist, key=lambda x: abs(x[1]), reverse=True)
            probabilities = softmax([abs(x[1]) for x in topic_dist])
            topic_probs = [(x[0], probabilities[i]) for i, x in enumerate(topic_dist)]
            return topic_probs
        else:
            # LDA结果已经是概率分布
            return sorted(topic_dist, key=lambda x: x[1], reverse=True)

    def get_word_topic_dict(self, topn=10):
        """
        获取主题对应的topn个关键词

        参数:
            topn: 返回的关键词数量

        返回:
            {主题ID: [(关键词, 权重)]}
        """
        for i in range(self.model.num_topics):
            self.word_topic_dict[i] = self.model.show_topic(i, topn)

    def predict_keyword(self, doc: str, key_n=20, topn=10):
        """
        根据文档的主题分布概率获取对应的关键词

        参数:
            doc: 文档
            key_n: 返回的关键词数量
            topn: 每个主题考虑的关键词数量

        返回:
            [关键词] 按权重排序后的列表
        """
        # 获取文档的主题分布
        topic_dist = self.predict_topic(doc=doc)

        # 获取每个主题的topn关键词及其权重
        self.get_word_topic_dict(topn=topn)

        # 初始化关键词得分字典
        keyword_scores = {}

        # 遍历文档的主要主题（使用所有主题而不仅仅是前几个，但可以加权重）
        for topic_id, topic_prob in topic_dist:
            if topic_id in self.word_topic_dict:
                topic_keywords = self.word_topic_dict[topic_id]

                # 计算每个关键词的加权得分,得分 = 文档在该主题的概率 * 关键词在该主题的概率
                for word, word_prob in topic_keywords:
                    score = topic_prob * word_prob

                    # 累加到总得分中
                    if word in keyword_scores:
                        keyword_scores[word] += score
                    else:
                        keyword_scores[word] = score

        # 按得分排序并返回topn关键词
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in sorted_keywords[:key_n]]

        return top_keywords


if __name__ == "__main__":
    d = DictCorpus()
    d = d.load_self()
    t = TopicModel(d)
    t.load()
    res = t.predict_keyword(load_pdf('public/金融危机历史镜鉴下的中国金融稳定立法_肖京.pdf'))
    print(res)

