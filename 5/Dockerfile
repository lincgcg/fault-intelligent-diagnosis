# 依赖的基础镜像(推荐官方提供的da-competition-base:1.0.0)
FROM 43.138.83.210/da/da-competition-base:1.0.0
# 将本地文件复制到容器中
RUN mkdir /app
COPY . /app
# 运行安装依赖包
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 设置工作路径
WORKDIR /app
# 容器启动后执行程序
CMD ["python","main.py"]