from PyQt5 import QtCore, QtGui, QtWidgets
from time import time

import models
import data_collections
import utils

models = {'Vector Space': models.VectorSpace, 'Extended Boolean': models.BooleanExtended}
models_s = {'Vector Space': 'vector_space', 'Extended Boolean': 'boolean_extended', 'Probabilistic': 'probabilistic'}
corpus = {'cran': data_collections.CranCollection, 'newsgroup': data_collections.NewsGroupCollection}

class Ui_MainWindow(object):
    def __init__(self):
        self.model = utils.deserialize('models/boolean_extended.pkl')
        self.corpus = ''
        self.query = None
        self.recovered = None
        self.relevant = []
        self.not_relevant = []
        self.groups = []
        self.relevants = []
        self.window = None
        self.temp = None
        self.page = 0
        self.rank = None
        self.rank_page = [[]]

    def setupUi(self, MainWindow):
        self.window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1024, 800)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.sys_name = QtWidgets.QLabel(self.centralwidget)
        self.sys_name.setGeometry(QtCore.QRect(400, 10, 300, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(20)
        self.sys_name.setFont(font)
        self.sys_name.setObjectName("sys_name")

        self.search_input = QtWidgets.QComboBox(self.centralwidget)
        self.search_input.setGeometry(QtCore.QRect(180, 50, 705, 31))
        self.search_input.setEditable(True)
        self.search_input.setObjectName("search_input")

        self.search_button = QtWidgets.QPushButton(self.centralwidget)
        self.search_button.setGeometry(QtCore.QRect(915, 50, 89, 31))
        self.search_button.setObjectName("search_button")

        self.page_next = QtWidgets.QPushButton(self.centralwidget)
        self.page_next.setGeometry(QtCore.QRect(535, 85, 25, 25))
        self.page_next.setObjectName("page_next")
        self.page_next.clicked.connect(self.forward)

        self.page_back = QtWidgets.QPushButton(self.centralwidget)
        self.page_back.setGeometry(QtCore.QRect(500, 85, 25, 25))
        self.page_back.setObjectName("page_back")
        self.page_back.clicked.connect(self.backward)

        self.corpus_combo = QtWidgets.QComboBox(self.centralwidget)
        self.corpus_combo.setGeometry(QtCore.QRect(45, 20, 131, 31))
        self.corpus_combo.setEditable(False)
        self.corpus_combo.setObjectName("corpus_combo")

        self.s_corpus = QtWidgets.QLabel(self.centralwidget)
        self.s_corpus.setGeometry(QtCore.QRect(5, 20, 40, 30))
        self.s_corpus.setObjectName("s_corpus")

        self.model_combo = QtWidgets.QComboBox(self.centralwidget)
        self.model_combo.setGeometry(QtCore.QRect(45, 60, 131, 31))
        self.model_combo.setEditable(False)
        self.model_combo.setObjectName("model_combo")

        self.s_model = QtWidgets.QLabel(self.centralwidget)
        self.s_model.setGeometry(QtCore.QRect(5, 60, 40, 30))
        self.s_model.setObjectName("s_model")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(12, 115, 1000, 655))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                           QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1000, 655))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.temp = QtWidgets.QLabel(self.centralwidget)
        self.temp.setGeometry(QtCore.QRect(12, 90, 200, 17))
        self.temp.setObjectName("Time")

        self.recovered = QtWidgets.QLabel(self.centralwidget)
        self.recovered.setGeometry(QtCore.QRect(200, 90, 200, 17))
        self.recovered.setObjectName("Recovered")

        self.in_page = QtWidgets.QLabel(self.centralwidget)
        self.in_page.setGeometry(QtCore.QRect(570, 90, 100, 17))
        self.in_page.setObjectName("In_Page")

        self.retranslateUi(MainWindow)
        self.corpus_combo.currentTextChanged['QString'].connect(self.set_corpus)
        self.model_combo.currentTextChanged['QString'].connect(self.set_model)
        self.search_button.clicked.connect(self.search_query)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def set_corpus(self, text):
        self.corpus = text
        self.query = None
        self.search_input.clear()

    def set_model(self, text):
        pre_model = models[text]
        path = 'models/' + models_s[text] + '_' + self.corpus + '.pkl'
        try:
            self.model = utils.deserialize(path)
        except FileNotFoundError:
            self.model = pre_model(corpus[self.corpus]().parse(), self.corpus)
            utils.serialize(self.model, path)
        self.query = None
        self.search_input.clear()

    def search_query(self):
        if self.search_input.currentText() != '':
            self.t = time()
            self.query = self.search_input.currentText()
            # repeat = 0
            # for q in self.SRI.get_querys():
            #     if q.get_text() == self.search_input.currentText():
            #         self.query = q
            #         repeat = 1
            #         break
            # if not repeat:
            #     self.query = self.SRI.create_query(self.search_input.currentText())
            #     self.SRI.insert_query(self.query)
            #     self.search_input.addItem(self.query._text)
            #
            # self.SRI.compare_query(self.query)
            # self.rank = (self.SRI.ranking(self.query)
            #              if len(self.query.get_relevants()) == 0
            #              else self.SRI.retro(self.query))
            self.rank = self.model.ranking(self.search_input.currentText(), 20)
            self.rank_page = [[] for i in range(int(len(self.rank) / 20) + 1)]
            # Con esta línea revisamos los ids de los documentos devueltos
            # print([i[0]._d_id for i in self.rank])
            element = 0
            page = 0
            for d in self.rank:
                if element < 19:
                    self.rank_page[page].append(d)
                    element += 1
                else:
                    element = 0
                    page += 1
                    self.rank_page[page].append(d)

            self.t = time() - self.t
            self.r = len(self.rank)
            self.show(self.rank_page[0])
            self.temp.show()

    def backward(self):
        if self.page > 0:
            self.page -= 1
            self.show(self.rank_page[self.page])

    def forward(self):
        if self.page < len(self.rank_page) - 1:
            self.page += 1
            self.show(self.rank_page[self.page])

    def show(self, r):
        if len(self.groups) > 0:
            self.groups = []
            self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
            self.scrollArea.setGeometry(QtCore.QRect(12, 115, 1000, 655))
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setObjectName("scrollArea")
            self.scrollAreaWidgetContents = QtWidgets.QWidget()
            self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1000, 655))
            self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
            self.scrollArea.setWidget(self.scrollAreaWidgetContents)
            self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
            self.verticalLayout.setObjectName("verticalLayout")
        index = 0
        for d in r:
            # self.query.set_not_relevant(d[0])
            g = self.group(self.scrollAreaWidgetContents, self.verticalLayout,
                           d, index, self.query)
            self.groups.append(g)
            index += 1
        self.scrollArea.show()
        self.retranslateUi(self.window)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Find All"))
        MainWindow.setWindowIcon(QtGui.QIcon('icon.ico'))
        self.sys_name.setText(_translate("MainWindow", "Find All - Search Engine"))
        self.search_button.setText(_translate("MainWindow", "Search"))
        self.page_back.setText(_translate("MainWindow", "<"))
        self.page_next.setText(_translate("MainWindow", ">"))
        self.s_corpus.setText(_translate("MainWindow", "Corpus:"))
        self.s_model.setText(_translate("MainWindow", "Model:"))
        self.in_page.setText(_translate("MainWindow", "Page " + str(self.page + 1) + '/' +
                                        str(len(self.rank_page))))
        if self.query != None:
            self.temp.setText(_translate("MainWindow", 'Time: ' + str(self.t)[:5] + ' seconds'))
            self.recovered.setText(_translate('MainWindow', 'Recovered: ' + str(self.r) + ' docs'))
        for group in self.groups:
            group.group.setTitle(_translate("MainWindow", "Document: " + group.subject))
            group.check.setText(_translate("MainWindow", "Relevant"))

    class group:
        def __init__(self, ScrollArea, layout, doc, index, query):
            self.area = ScrollArea
            self.doc = doc
            self.subject = doc.title
            self.index = index
            self.query = query

            self.group = QtWidgets.QGroupBox(self.area)
            self.group.setGeometry(QtCore.QRect(10, 10 + 101 * index, 741, 150))
            self.group.setObjectName(self.subject)
            layout.addWidget(self.group)
            self.my_layout = QtWidgets.QVBoxLayout(self.group)
            self.my_layout.setObjectName("layout" + str(index))
            self.group.show()

            self.check = QtWidgets.QCheckBox(self.group)
            self.check.setObjectName("relevant_" + str(index))
            self.check.setGeometry(QtCore.QRect(640, 30, 92, 23))

            # if doc in query.get_relevants():
            #     self.check.setChecked(True)
            self.check.setChecked(False)

            self.my_layout.addWidget(self.check)
            # self.check.stateChanged.connect(lambda: self.relevant(self.check))
            self.check.show()

            self.text = QtWidgets.QTextEdit(self.group)
            self.text.setEnabled(True)
            self.text.setGeometry(QtCore.QRect(0, 19, 631, 81))
            self.text.setFrameShape(QtWidgets.QFrame.Box)
            self.text.setFrameShadow(QtWidgets.QFrame.Plain)
            self.text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.text.setObjectName("text_" + str(index))
            self.text.setText(doc.corpus)
            self.text.setReadOnly(True)
            self.text.setFrameShape(QtWidgets.QFrame.HLine)
            self.my_layout.addWidget(self.text)
            self.text.show()

        def relevant(self, c):
            if c.isChecked():
                self.query.set_relevant(self.doc)
                if self.doc in self.query.get_not_relevants():
                    self.query.get_not_relevants().remove(self.doc)
            else:
                self.query.set_not_relevant(self.doc)
                if self.doc in self.query.get_relevants():
                    self.query.get_relevants().remove(self.doc)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.corpus_combo.addItems(['cran', 'newsgroup'])
    ui.model_combo.addItems(['Vector Space', 'Extended Boolean', 'Probabilistic'])
    MainWindow.show()
    sys.exit(app.exec_())
