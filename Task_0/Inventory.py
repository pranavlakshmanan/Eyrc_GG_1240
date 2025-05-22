T = int(input())
for i in range(0, T):
    L=[[],[]]
    item_list = L[0]
    quantity_list = L[1]
    item_name = ""
    quantity = 0
    N = int(input())
    for j in range(0, N):
        x, y = [x for x in input().split(" ")]
        item_list.append(x)
        quantity_list.append(int(y))
    M = int(input())
    for j in range(0, M):
        operation_name = input()
        if (operation_name[0:3].upper() == "ADD"):
            item_name = operation_name[4:operation_name.rindex(" ")]
            quantity = int(
                operation_name[operation_name.rindex(" "):len(operation_name)])
            if item_name in item_list:
                v = item_list.index(item_name)
                quantity_list[v] += quantity
                print("Updated Item", item_name)
            if item_name not in item_list:
                item_list.append(item_name)
                quantity_list.append(quantity)
                print("ADDED Item", item_name)
        if (operation_name[0:6].upper() == "DELETE"):
            item_name = operation_name[7:operation_name.rindex(" ")]
            quantity = int(
                operation_name[operation_name.rindex(" "):len(operation_name)])
            if item_name in item_list:
                v = item_list.index(item_name)
                if (quantity_list[v] < quantity):
                    print("Item", item_name, "could not be DELETED")
                if (quantity_list[v] > quantity):
                    quantity_list[v] -= quantity
                    print("DELETED Item", item_name)
                if(quantity_list[v] == quantity):
                    quantity_list.remove(quantity)
                    item_list.remove(item_name)
                    print("DELETED Item", item_name)
            if item_name not in item_list:
                print("Item", item_name, "does not exists")

    print("Total Items in Inventory:", sum(quantity_list))