class CookieMaker:
    def __init__(self):
        self.result = 0   # 초기 입력 값 설정 (0값을 가지는 result라는 변수 만듦)

    def add(self, num):   # CookieMaker가 가지는 함수
        self.result += num
        return self.result

    def reset(self):
        self.result =0
        return self.result

a = list()  # list() : class , a : instance
a.append(3)


cookie = CookieMaker()  # CookieMaker() : class , cookie : instance
cookie.result
cookie.add(3)
cookie.result

CookieMaker.add(cookie,4)
cookie.result

cookie.add(5)
cookie.result