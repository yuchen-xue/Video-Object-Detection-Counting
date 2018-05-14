lst = []
# set_a = set(map(lambda x: (x, lst.count(x)), lst))
# # for item in set_a:
# #     print(item)
#
# new_lst = [x for x in set_a]
# for item in new_lst:
#     print('{}, {}'.format(item[0], item[1]))

for i, (item1, item2) in enumerate(x for x in set(map(lambda x: (x, lst.count(x)), lst))):
    print('{}: {}, {}'.format(i, item1, item2))

