# !pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# px 진행


# x: bill_length_mm
# y: bill_depth_mm  
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species"
)

fig.update_layout(
    title={'text' : "<span style = 'color:blue; font-weight:bold;'> 팔머펭귄 </span>",
           'x' : 0.5,  # 왼쪽0, 오른쪽 1을 기준으로 0.5위치에 둬라는 것임. 가운데 정렬
           'xanchor' : 'center',  # 이거 변화 없는 것 같은데...
           'y' : 0.9}  # 맨 아래 0, 맨 위에 1을 기준으로 0.5위치에 둬라는 것임. 가운데 정렬
) 
fig.show() # vscode에서는 그냥 fig만 해도 그림 그려짐 (근데 fig 안해도 그려지는 것 같음)

"""
<span> ... </span>  # <span>은 청크의 시작  #</span>은 청크의 끝

<span>
<span> ... </span>   # 하위 청크도 가능
"<span style = 'font-weight:bold'> ...해당 내용 </span> "   # 글씨 굵게 설정  # 글씨니까 큰 따옴표로 싸줘야함.
"<span style = 'color:blue; font-weight:bold; '> ...해당 내용 </span> "   # 같은 내용에 대해서 설정을 여러개 해주고 싶을 때

</span>
"""

#style = 이라는 설정에 있는 설정들
#color, font-weight, font-size