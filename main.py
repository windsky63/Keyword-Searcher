import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5 import QtWidgets
from ui.main import Ui_Form
from model import DictCorpus, TopicModel
from data_load import load_pdf, load_docx, load_pic
import os
from typing import Union


class Mainwindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        # 设置UI
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 初始化变量
        self.selected_files = []
        self.selected_folder = ""
        self.corpus_files = []
        self.corpus_folders = []

        # 绑定事件和槽
        self.bind_events()

        # 初始化UI状态
        self.init_ui_state()

    def bind_events(self):
        """绑定UI事件和槽函数"""
        # 文件选择页按钮
        self.ui.pushButton_file.clicked.connect(self.select_file)
        self.ui.pushButton_folder.clicked.connect(self.select_folder)

        # 模型训练页按钮
        self.ui.pushButton_retrain.clicked.connect(re_train)

        # 语料管理页按钮
        self.ui.pushButton_add_file.clicked.connect(self.add_corpus_file)
        self.ui.pushButton_add_folder.clicked.connect(self.add_corpus_folder)

        # 主题数量滑块和微调框同步
        self.ui.horizontalSlider_topics.valueChanged.connect(
            self.ui.spinBox_topics.setValue)
        self.ui.spinBox_topics.valueChanged.connect(
            self.ui.horizontalSlider_topics.setValue)

    def init_ui_state(self):
        """初始化UI状态"""
        # 设置默认模型为LSI
        self.ui.comboBox_model.setCurrentIndex(0)

        # 初始化状态栏
        self.ui.statusBar.showMessage("就绪")

    def select_file(self):
        """选择文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文件", "",
            "支持的文件 (*.docx *.pdf *.jpg *.png)")

        if files:
            self.selected_files.extend(files)
            self.ui.statusBar.showMessage(
                f"已选择 {len(files)} 个文件", 3000)
            # 这里可以添加文件显示到表格的逻辑

    def select_folder(self):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.selected_folder = folder
            self.ui.statusBar.showMessage(
                f"已选择文件夹: {folder}", 3000)

    def add_corpus_file(self):
        """添加语料文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "添加语料文件", "",
            "支持的文件 (*.docx *.pdf *.jpg *.png *.txt)")

        if files:
            self.corpus_files.extend(files)
            self.ui.statusBar.showMessage(
                f"已添加 {len(files)} 个语料文件", 3000)

    def add_corpus_folder(self):
        """添加语料文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "添加语料文件夹")
        if folder:
            self.corpus_folders.append(folder)
            self.ui.statusBar.showMessage(
                f"已添加语料文件夹: {folder}", 3000)

    def extract_keywords(self):
        """提取关键词"""
        if not self.selected_files and not self.selected_folder:
            QMessageBox.warning(
                self, "警告", "请先选择要提取关键词的文件！",
                QMessageBox.Ok, QMessageBox.Ok)
            return

        self.ui.statusBar.showMessage("正在提取关键词...", 0)

        # 示例数据 - 实际应用中应该从模型获取
        sample_data = [
            ["文件1.docx", "关键词1, 关键词2, 关键词3", "0.95"],
            ["文件2.pdf", "关键词A, 关键词B", "0.92"],
            ["图片1.jpg", "风景, 自然, 户外", "0.88"]
        ]

        # 设置表格数据
        self.setup_table_view(sample_data)
        self.ui.statusBar.showMessage("关键词提取完成！", 3000)

    def setup_table_view(self, data):
        """设置表格视图数据"""
        # 清除现有数据
        self.ui.tableView.setRowCount(0)
        self.ui.tableView.setColumnCount(0)

        # 设置列数和标题
        self.ui.tableView.setColumnCount(3)
        self.ui.tableView.setHorizontalHeaderLabels(
            ["文件名", "关键词"])

        # 添加数据
        for row_num, row_data in enumerate(data):
            self.ui.tableView.insertRow(row_num)
            for col_num, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.ui.tableView.setItem(row_num, col_num, item)

        # 调整列宽
        self.ui.tableView.resizeColumnsToContents()


def predict(path: str):
    """
    预测给定路径指向的文件或文件夹中的文档主题关键词

    参数:
        path: 文件路径(PDF/DOCX/图片)或文件夹路径

    返回:
        预测结果列表，每个元素是(文件路径, 预测结果)的元组
    """
    d = DictCorpus()
    d = d.load_self()
    t = TopicModel(d)
    t.load()

    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")

    results = []

    # 如果是文件夹路径
    if os.path.isdir(path):
        # 获取文件夹内所有支持的文件
        supported_extensions = ('.pdf', '.docx', '.png', '.jpg', '.jpeg')
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
                try:
                    doc_content = load_file(file_path)
                    if doc_content is not None:
                        res = t.predict_keyword(doc=doc_content)
                        results.append((file_path, res))
                        print(f"处理完成: {filename} -> {res}")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                    continue
    # 如果是单个文件路径
    elif os.path.isfile(path):
        file_path = path
        try:
            doc_content = load_file(file_path)
            if doc_content is not None:
                res = t.predict_keyword(doc=doc_content)
                results.append((file_path, res))
                print(f"预测结果: {res}")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    return results


def load_file(file_path: str) -> Union[str, None]:
    """
    根据文件扩展名加载文件内容

    参数:
        file_path: 文件路径

    返回:
        文件内容文本或图片特征向量，出错时返回None
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.pdf':
            return load_pdf(file_path)
        elif ext == '.docx':
            return load_docx(file_path)
        elif ext in ('.png', '.jpg', '.jpeg'):
            return load_pic(file_path)
        else:
            print(f"不支持的文件格式: {ext}")
            return None
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None


def add_corpus():
    try:
        d = DictCorpus()
        d.load_from_pdf()
        d.to_dict()
        d.document_to_vector()
        d.save_self()
    except Exception as e:
        print(e)


def re_train(model_type="LSI", num_topics=20):
    try:
        d = DictCorpus()
        d.load_self()

        topic_model = TopicModel(d, model=model_type, num_topics=num_topics)
        topic_model.train()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Mainwindow()
    w.show()
    sys.exit(app.exec_())
