class Animal(object):

    def __init__(self, inputs):
        self.inputs = inputs

    # __call__(self, words):
    #     print ("Hello: ", words)

    def _call(self):
        print("Hello: ", self.inputs)

if __name__ == "__main__":
    cat = Animal("I am cat!")