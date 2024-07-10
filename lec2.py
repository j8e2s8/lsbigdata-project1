# 실행 ctrl+enter
# 블록 shift+ 화살표

a=1
a
a

a=10 # assign 할당 
a
a = "안녕하세요!"
a
a = '안녕하세요!'
a

a=[1,2,3]
a
b = [4,5,6]
a+b

a='안녕하세요'
b='LS 빅데이터 스쿨'
a+b

a+' '+b

print(a)

print("a+b =",' ', a+b)
print("a+b= ",a+b)

a=5
b=2
print("a-b= ", a-b)

print("a*b=", a*b)

print("a/b=", a/b) 

a == b
a != b
a<b
a>b
a<=b
a>=b


a=(2**4 + 12453//7)%8
b=(9**7/12) * (36452%253)
a<b


user_age =14
is_adult = user_age >= 18
print("성인입니까?", is_adult)

# False, True : dPdirdjdla
False = 3
True =2

a="True"  # 문자열이라서 잘 됨
b=TRUE   # 만든 적 없는 변수명이 지정되어서
c=true   # 만든 적 없는 변수명이 지정되어서
d=True   # 예약어 True임.

TRUE = 4
b=TRUE
b

# True, False
a=True
b=False

a and b
a or b

# and 연산자 : *과 같음
True and False # Fasel
True and True #True
False and False # False
False and True # False

True * False # Fasel 0
True * True #True 1
False * False # False 0
False * True # False 0 

# or 연산자 : +과 같음 (0은 false, 1이상은 true인 꼴), 정확히는 min(a+b,1)
True or False #True
False or False #False
False or True #True
True or True #True

True +False #True
False+ False #False
False+ True #True
True +True #True

min(True +False,1) #True 1
min(False+ False,1) #False 0
min(False+ True,1) #True 1
min(True +True,1) #True 1

# True : 1, False :0
True + True # 2
True + False # 1
False + False # 0


#복합 연산
a=3
a += 10
a

a -= 4
a

a *= 2
a

a /= 3
a


# 문자열 반복
str1 = "hello!"
repeated_str = str1 *3
print("Repeated string:", repeated_str)


# 단항 연산자 (부호를 설정하고 싶을 때)
x=5
+x
-x
~x # 비트 표현(2진수)를 반전시킨 값

y=-5
+y
-y
~y

# 비트
bin(5) # 5를 어떻게 처리하고 있니? bin = binary #문자열로 처리하고 있음 '0b'는 2진수로 표현한다는 뜻임.
bin(-5)

x=3
bin(3)
bin(~x)

x=-2
bin(~x)

x=-3
bin(~x)

~x


-128+64+32+16+8+2
x=5
bin(x)
bin(~x)

x=-3
bin(~x)
x=6
~x


-128+64+32+16+8+4+2+1
x=0
bin(~x)
~x
-128+64+32+16+8+2
x=5
~x

x=0
bin(~x)


pip install pydataset  # 문법 오류 뜸


import pydataset
pydataset.data()

df = pydataset.data('AirPassengers')
df


!pip install pandas
