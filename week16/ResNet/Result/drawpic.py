import matplotlib.pyplot as plt

epoch = list(range(200))


def get_data(file_name):
    train_acc = []
    test_acc = []
    with open(file_name) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.strip().startswith('*'):
                # print(line)
                # print(float(line[-7:]))
                test_acc.append(float(line[-7:]))

            elif line.strip().startswith('Epoch') and lines[i + 1].strip().startswith('Test'):
                # print(line)
                # print(float(line[-8:-2]))
                train_acc.append(float(line[-8:-2]))
    return train_acc, test_acc


train20, test20 = get_data('log_resnet20')
train32, test32 = get_data('log_resnet32')
train44, test44 = get_data('log_resnet44')
train56, test56 = get_data('log_resnet56')

# lr_train_01, lr_test_01 = get_data('log_0.1')
# lr_train_02, lr_test_02 = get_data('log_0.2')
# lr_train_005, lr_test_005 = get_data('log_0.05')
# lr_train_05, lr_test_05 = get_data('log_0.5')
# lr_train_001, lr_test_001 = get_data('log_0.01')
#
# b_train_64, b_test_64 = get_data('log_b_64')
# b_train_128, b_test_128 = get_data('log_0.1')
# b_train_512, b_test_512 = get_data('log_b_512')
# b_train_1024, b_test_1024 = get_data('log_b_1024')
# b_train_2048, b_test_2048 = get_data('log_b_2048')

fig = plt.figure()
sub = fig.add_subplot(111)
sub.plot(epoch, train20, label='resnet20')
sub.plot(epoch, train32, label='resnet32')
sub.plot(epoch, train44, label='resnet44')
sub.plot(epoch, train56, label='resnet56')
sub.legend()
sub.set_xlabel('epoch')
sub.set_ylabel('train accuracy')
plt.ylim(80, 100)

fig2 = plt.figure()
sub2 = fig2.add_subplot(111)
sub2.plot(epoch, test20, label='resnet20')
sub2.plot(epoch, test32, label='resnet32')
sub2.plot(epoch, test44, label='resnet44')
sub2.plot(epoch, test56, label='resnet56')

sub2.legend()
sub2.set_xlabel('epoch')
sub2.set_ylabel('test accuracy')

plt.ylim(80, 100)
# ==================================
# fig = plt.figure()
# sub = fig.add_subplot(111)
# sub.plot(epoch, train20, label='resnet20')
# sub.plot(epoch, test20, label='resnet20', linestyle='--')

# fig = plt.figure()
# sub = fig.add_subplot(111)
# sub.plot(list(range(100)), lr_train_05, label='lr=0.5')
# sub.plot(list(range(100)), lr_train_02, label='lr=0.2')
# sub.plot(list(range(100)), lr_train_01, label='lr=0.1')
# sub.plot(list(range(100)), lr_train_005, label='lr=0.05')
# sub.plot(list(range(100)), lr_train_001, label='lr=0.01')
# sub.legend()
# sub.set_xlabel('epoch')
# sub.set_ylabel('train accuracy')

# fig = plt.figure()
# sub = fig.add_subplot(111)
# sub.plot(list(range(100)), lr_test_05, label='lr=0.5')
# sub.plot(list(range(100)), lr_test_02, label='lr=0.2')
# sub.plot(list(range(100)), lr_test_01, label='lr=0.1')
# sub.plot(list(range(100)), lr_test_005, label='lr=0.05')
# sub.plot(list(range(100)), lr_test_001, label='lr=0.01')
# sub.legend()
# sub.set_xlabel('epoch')
# sub.set_ylabel('test accuracy')
# =============================================

# fig2 = plt.figure()
# sub2 = fig2.add_subplot(111)
# sub2.plot(list(range(100)), b_train_64, label='batch size=64')
# sub2.plot(list(range(100)), b_train_128, label='batch size=128')
# sub2.plot(list(range(100)), b_train_512, label='batch size=512')
# # sub2.plot(list(range(100)), b_train_1024, label='batch size=1024')
# sub2.plot(list(range(100)), b_train_2048, label='batch size=2048')
#
# sub2.legend()
# sub2.set_xlabel('epoch')
# sub2.set_ylabel('train accuracy')

# fig2 = plt.figure()
# sub2 = fig2.add_subplot(111)
# sub2.plot(list(range(100)), b_test_64, label='batch size=64')
# sub2.plot(list(range(100)), b_test_128, label='batch size=128')
# sub2.plot(list(range(100)), b_test_512, label='batch size=512')
# # sub2.plot(list(range(100)), b_test_1024, label='batch size=1024')
# sub2.plot(list(range(100)), b_test_2048, label='batch size=2048')
#
# sub2.legend()
# sub2.set_xlabel('epoch')
# sub2.set_ylabel('test accuracy')

# plt.ylim(80, 100)
# plt.ylim(80, 100)
plt.show()
