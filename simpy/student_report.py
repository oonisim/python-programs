import simpy

env = simpy.Environment()


def student(env, store):
    for i in range(3):
        yield env.timeout(15)
        print('student reporting at %d' % (env.now))
        store.put("report#%d" % i)


def teacher(env, store):
    while True:
        yield env.timeout(5)
        print('teacher waits for student to report at %d' % (env.now))

        report = yield store.get()
        print("teacher got student report at %d" % (env.now))


store = simpy.Store(env)
s = env.process(student(env, store))
t = env.process(teacher(env, store))
env.run(until=60)
