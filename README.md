# gcn-tensorflow代码学习记录

* 代码来自: https://github.com/tkipf/gcn

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
没有完全懂GCN相关内容，但大体跑通了这份代码，相关记录如下。

## 数据

* 根据作者README.md中所述，当需要加载自定义数据时，需要提供三个矩阵：

    * adjacent_m: N*N的邻接矩阵
    
    * fetures_m: N*D的节点特征向量矩阵
    
    * res_m: N*E的矩阵，E是节点所属种类的数量
    
    * 以上参数中，N是节点数量，D是节点特征维度，E是节点种类数量

* 于是对作者所给的data进行研究，以cora数据为例:

    * 关于cora数据的描述，可参见https://linqs.soe.ucsc.edu/data。原文对数据的描述是：
    
    ```
      The Cora dataset consists of 2708 scientific publications classified into one of seven classes. 
      The citation network consists of 5429 links. 
      Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. 
      The dictionary consists of 1433 unique words.  
    ```
    
    * 也就是说，数据集中包括2708个分别属于7个种类的科学出版商；整个数据集包括1433个不重复的word,每个publication是基于1433个word的onehot编码来表示的。
    
    * 具体运行时，将train.py的dataset参数值置为'cora'即可。
    
* 关于./data目录下数据读取、含义和预处理如下：
    
    * 作者在README.md中说，数据读取参考了https://github.com/kimiyoung/planetoid的处理，具体见utils.py中的load_data()函数。首先对不同后缀文件含义进行解释：
    
        * x,y后缀数据文件是训练数据中已标注的节点features和对应标签的one-hot编码
        * tx,ty后缀数据文件测试数据中已标注的节点features和对应标签的one-hot编码
        * allx,ally是已标注和未标注训练数据数据的features和标签的one-hot编码 (ally容量和allyx容量相同,那不就是都有标注了吗?)。
        * graph是以字典形式给出的每个节点及其邻接节点的表示(每个节点用Index表示)
        * test后缀是测试集数据节点的index
    
    * graph数据读取后，会通过networkx.adjacency_matrix生成对应的邻接矩阵。
    
    * 其他具体数据维度见utils.py中的注释
    * data_content.txt中是读取数据的部分结果
    
* **注意，当要使用自定义数据时，提供节点的features表示(例如，作者这里的处理是One-hot)，label的one-hot表示并划分好训练集、验证集和测试集，仿照load_data()函数进行处理，即可送入模型**。
                

## Model

### model.py

* model.py中定义了基本的模型架构，具体见build函数。

* **self.output**的结果见最原始Model中的定义(但是没有看懂)。output后的self.preds是自己为了方面后续查看结果添加的。

* 卷积层的添加来自layer.py


### layer.py

layer.py中提供了具体**图卷积层**的写法[然而也没有完全看懂]


## 运行

python train.py即可得到每个epoch的结果。

## 其他总结

* 虽然很多细节没有看懂，但是学会了如何使用这份代码。以后需要的话，将数据处理成所需要的形式即可调用。

* 这份代码的Pytorch版本，见 https://github.com/tkipf/pygcn

* 图神经网框架[DGL](https://docs.dgl.ai/install/index.html)也提供了GCN相关接口

* Pytorch图神经网络框架[PyG](https://github.com/rusty1s/pytorch_geometric)也提供了GCN接口