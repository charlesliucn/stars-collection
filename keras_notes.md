 # Keras笔记
## 1. 关于Python中对向量/张量中axis的理解
+ 在numpy, tensorflow, keras等诸多python库中，都定义了张量的axis
+ axis理解为维度的角标
+ 类似于sum, max函数，axis=n则意味着结果中的第n维消失，也就是说对第n维进行了降维操作。
+ 举例：a.shape = [m,n,p,q]，则：
	a.sum(axis = 0).shape = [1,n,p,q]
	a.sum(axis = 1).shape = [m,1,p,q]
	...
+ axis=-1表示张量的最后一个维度
+ 与sort相关的axis, 分为以下情况:
	axis = None, 表示将张量flatten为一维的之后进行排序
	axis = n, 表示沿着第n维度看去，其他维度是从小到大有序排列的

## 2. keras中model的常用method方法：
    model.summary()
    model.get_config():返回模型配置信息的字典
    model.get_layer()
    model.get_weights()
    model.set_weights()
    model.to_json() ←→ model_from_json()
    model.to_yaml() ←→ model_from_yaml()
    model.save_weights()
    model.load_weights()

## 3. keras中Sequential模型的接口：
    model = keras.models.Sequential()
    model.add(...)
    model.pop(): 弹出模型的最后一层
    model.compile()
    model.fit():validation_split的划分在shuffle之前，所以如果数据本身有序，则需要先手动shuffle，然后在validation_split；因为model.fit()的shuffle是只针对切分后所得的训练集生效的。
    model.evaluate()
    model.predict():输出预测的类别
    model.predict_proba():输出预测类别的概率
    model.train_on_batch():在一个batch的样本上进行训练
    model.test_on_batch():在一个batch的样本上进行评估
    model.predict_on_batch():在一个batch的样本上进行预测
    model.fit_generator()
    model.evaluate_generator()
    model.predict_generator()

## 4. keras中Model模型的接口：
    model = Model()
    model.layers
    model.inputs
    model.outputs

## 5. keras获取某层的输出：
model = LSTM(32)
单个输出：model.output
多个输出：model.get_output_at(1) ...

## 6. keras常用的layer(keras.layers(实际上是keras.layers.core))
    Dense(·)
    Activation(·)
    Dropout(·)
    Flatten(·):常用于卷积层到全连接层的过渡
    Reshape(·):将输入转换为特定的shape，指定shape时，可以使用-1代替其中一个维度
    Permute(·):将输入张量按指定模式进行维度的重排
    RepeatVector(·): 将输入重复若干次
    Lambda(·):用户自定义的函数
    ActivityRegularizer(·):数据不会发生任何变化，但会基于激活值更新损失函数
    Masking(·):指定mask_value，当张量某一维度的数值均为该数值时，则会被屏蔽，之后的网络层都看不到mask部分的数据

## 7. 用于卷积层的常见对象参数：
    Initializer():kernel_initializer, bias_initializer
    Regularizer():kernel_regularizer, bias_regularizer, activity_regularizer
    Constraints():kernel_constraints, bias_constraints

## 8. 常用的功能层：
### 1) 卷积层(layers.convolutional)
	Conv1D:一维卷积
	Conv2D:二维卷积
	SeparableConv2D:可分离卷积，加上:
		depthwise_(regularizer/constraint)
		pointwise_(regularizer/constraint)
	Conv2DTranspose:二维反卷积
	Conv3D:三维卷积
	
	Cropping1D:一维裁剪，需指定两端要裁剪的元素数
	Cropping2D:二维裁剪，需指定宽和高方向上两端裁剪的元素数
	Cropping3D:二维裁剪，三个维度上需要裁剪的元素数
	
	UpSampling1D:对一维数据上采样，相当于对元素重复size次
	UpSampling2D:对二维数据上采样，相当于对元素在两个方向上分别重复size[0]和size[1]次
	UpSampling3D:对三维数据上采样，同上原理
	
	ZeroPadding1D:一维填充0
	ZeroPadding2D:二维填充0，可在不同维度填充不同数量的0
	ZeroPadding3D:三维填充0

### 2) 池化层(layers.pooling)
	MaxPooling1D
	MaxPooling2D
	MaxPooling3D
	AveragePooling1D
	AveragePooling2D
	AveragePooling3D
	GlobalMaxPooling1D
	GlobalAveragePooling1D
	GlobalMaxPooling2D
	GlobalAveragePooling2D

### 3) 局部连接层(layers.local)
	LocallyConnected1D:与Conv1D的区别在于，不进行权值共享，不同位置的滤波器是不一样的
	LocallyConnected2D

### 4) 循环层(layers.recurrent)
	Recurrent:抽象类，无法实例化任何对象
	SimpleRNN:recurrent_(regularizer, initializer, constraints, dropout)
	GRU:门限循环单元
	LSTM:长短期记忆模型

### 5) 嵌入层(layers.embeddings)
	Embedding:只能作为模型的第一层
		embeddings_(initializer,regularizer,constraint)

### 6) 融合层(layers.merge)
	merge.Add 			------>		add
	merge.Multiply		------>		multiply
	merge.Average 		------> 	average
	merge.Maximum   	------> 	maximum
	merge.Concatenate	------>		concatenate
	merge.Dot 			------>		dot

### 7) 高级激活层(layers.advanced_activations)
	LeakyReLU: 		修正线性单元	alpha代表第三象限的斜率
	PReLU:			参数线性单元	alpha是可变的(参数化的)，有alpha_(initializer,regularizer,constraint)
	ELU:			指数线性单元	alpha为参数
	ThresholdedReLU:带有门限的ReLU theta为参数，代表门限

### 8) 规范化层(layers.normalization)
	BatchNormalization:	在每个batch上将前一层激活值重新规范化，使得均值接近于0，标准差接近于1
		超参数：		momentum, epsilon
		训练参数：	{beta, gamma}_{initializer, regularizer}, {moving_mean, moving_variance}_initializer

### 9) 噪声层(layers.noise)
	GaussianNoise: 为数据加噪(加性高斯噪声)，克服过拟合
		去噪自动编码器，试图从加噪输入中重构无噪声信号
	GaussianDropout: 正则化层，在训练时才有效

### 10) 封装层(layers.wrappers)
	TimeDistributed: 把一个层应用到输入的每个时间步上，输入至少为3D张量
	Bidirectional: 双向RNN的封装器
		merge_mode：前向和后向RNN输出的结合方式，默认为concat拼接，其余还包含：
			sum、mul、concat、ave、None

### 11) 自定义层
	简单的自定义层可以使用layers.core.Lambda
	复杂的自定义层需要定义类，并包含是三个方法(method):
		build(input_shape), call(x), compute_output_shape(input_shape)

## 9. 数据预处理(keras.preprocessing)
### 1) 序列预处理(keras.preprocessing.sequence)
	pad_sequences(): 指定maxlen后，可以对序列进行裁剪(truncating)或补齐(padding)，补齐还可以指定补齐的value
			注意函数在执行内部操作时，是先进行裁剪，然后进行补齐
	skipgrams(): 提取跳字样本
	make_sampling_table:生成序列抽样概率表，获得的是抽样概率表，在文本预料中出现概率越高的词，对应的采样频率应该越低

### 2) 文本预处理(keras.preprocessing.text)
	text_to_word_sequence: 将一个句子拆分成单词构成的序列
	one_hot: 将一段文本编码为整数数组，每个整数编码一个词(唯一性无法保证)
	Tokenizer类：用于向量化文本的类
	方法
		fit_on_texts: 输入用于训练文本列表
		texts_to_sequences: 输入待转为序列的文本列表
		texts_to_sequences_generator:上面函数的生成器版本
		texts_to_matrix:输入待向量化的文本，及向量化的方式
		fit_on_sequences: 输入要训练的序列列表
		sequences_to_matrix: 输入待向量化的序列列表
	属性
		word_counts: 字典形式，每个单词对应出现的次数
		word_docs: 字典形式，每个单词出现的文档数
		word_index: 单词索引
		document_count: 文档数目

### 3) 图像预处理(keras.preprocessing.image)
	ImageGenerator类：用于不断生成一个batch的数据
  	支持数据增强：均值化、白化；图像的旋转、平移、伸缩与翻转，防止过拟合
  	参数很多
  	方法
  		fit:	均值化、白化时需要使用此函数
  		flow:	不断返回batch数据
  		flow_from_directory:	在无限循环中产生batch数据

## 10. Keras内置的关于网络的相关配置
	对于一个深度学习项目，需要系统考虑以下几个因素：

### 1)	losses:		损失函数		(keras.losses)
		常用的损失函数
		mae, mse
		{binary,categorical,spare_categorical}_{crossentropy}

### 2)	metrics:	模型评估指标
		常用的模型评估指标	(keras.metrics)
		一般分类:	binary_accuracy, binary_crossentropy
					categorical_accuracy, categorical_crossentropy
		稀疏分类:	sparse_categorical_accuracy
					sparse_categorical_crossentropy
		模糊评估:	top_k_categorical_accuracy

### 3)	optimizer:	优化方法		(keras.optimizers)
		SGD:		随机梯度下降
		Adagrad:	自适应梯度下降
		Adadelta:	对Adagrad的扩展
		RMSprop:	均方根后向传播，对于RNN效果较好
		Adam:		自适应动量估计，目前在各方面表现都较好
		Adamax:		Adam的衍生版
		Nadam:		Adam的衍生版
		常选择：RMSprop, Adam和Nadam的某一种优化方法

### 4)	activation:	激活函数		
		一般激活函数(keras.activations):	常用softmax, ReLU, sigmoid和tanh
			softmax
			ReLU
			ELU
			softplus
			Sigmoid
			softsign
			tanh: 收敛比sigmoid更快
			linear
		高级激活函数(keras.advanced_activations):
			LeakyReLU
			PReLU
			ELU
		可以使用keras.backend可以自定义激活函数

### 5)	initializer:初始化方法	(keras.initializers)
		Zeros: 			全0初始化，效果极差，慎用
		Ones:			全1初始化
		Constant:		固定值初始化
		RandomNormal:	正态分布随机初始化
		RandomUniform:	均匀分布随机初始化
		TrunctedNormal:	截尾高斯分布初始化(限定在2σ范围以内)
		VarianceScaling:方差缩放初始化
		Orthogonal:		随机正交矩阵的初始化(正交矩阵的乘性系数)
		Identity:		二维矩阵的初始化(乘性系数)
		lecun_uniform:	LeCun均匀分布初始化方法
		glorot_uniform:	Glorot/Xavier均匀分布初始化
		glorot_normal:	Glorot/Xavier正态分布初始化
		He_uniform:		He均匀分布初始化
		he_normal:		He正态分布初始化
		可以使用keras.backend可以自定义初始化方法

### 6)	regularizer:正则项	(keras.regularizers)
		Regularizer正则对象
		l1:		L1范数正则化
		l2:		L2范数正则化
		l1_l2:	L1和L2范数正则化同时
		可以使用keras.backend可以自定义正则项

### 7)	constraint:	约束项	(keras.constraints)
		max_norm:		最大范数(模)约束
		non_neg:		非负性约束
		unit_norm:		单位范数约束，强制张量最后一个axis拥有单位范数
		min_max_norm:	强制norm在一个区间内

## 11. keras的回调(keras.callbacks)
	keras使用回调观察训练过程网络内部的状态和统计信息，传递到模型的fit中

### 1) Callback():
	on_{train,epoch,batch}_{begin,end}:共六种调用方式
### 2) BaseLogger:
	在每个epoch累加性能评估，在每个keras模型都会被自动调用
### 3）ProgbarLogger:
	将metrics指定的监视指标输出，也会被自动调用
### 4) History:
	model.fit()的返回值，返回一些历史的汇总信息
### 5) ModelCheckpoint:
	保存模型到filepath，默认格式为科学数据常用的hdf5格式
### 6) EarlyStopping:
	检测值不再改善时，提前终止训练
### 7) RemoteMonitor:
	想服务器发送事件流
### 8) LearningRateScheduler:
	参数是一个关于epoch号的函数，该函数返回值对应每个epoch设置的学习率
### 9) TensorBoard:
	可视化工具，根据参数设定可视化要求后，作为model.fit/fit_generator中参数callbacks的值即可
### 10)	ReduceLROnPlateau:
	评价指标不再提升时，减少学习率
### 11) CSVLogger:
	将每次epoch训练结果保存在csv中
### 12) LambdaCallback:
	创建简单的callback的匿名函数

## 12. keras模型的保存与恢复
### 1) 模型的保存
	model = Model() #某个模型
	model.save(filepath)
	model.save_weights()
	model.to_json()
	model.to_yaml()

### 2) 模型的恢复(keras.models)
	load_model(filepath)
	load_weights()
	model_from_json()
	model_from_yaml()

## 13. keras中scikit-learn的接口
	keras.wrappers.scikit-learn.KerasClassifier: 分类器的接口
	keras.wrappers.scikit-learn.KerasRegressor:	 回归器的接口
	KerasClassifier和KerasRegressor都可以使用sklearn的GridSearchCV等参数优化函数

## 14. keras内置可视化工具

### 1) plot_model
	from keras.utils import plot_model
	plot_model(model, to_file = "model.png") # show_shapes, show_layer_names

### 2) model_to_dot
	form keras.utils.vis_utils import model_to_dot
	model_to_dot(model).create(prog = "dot", format = "svg") # 得到的是pydot.Graph对象
