docs = []

for i in range(1000):
    doc = {"_id": str(i), "title": "title" + str(i), "content": "content" + str(i)}
    for j in range(200):
        doc[f"field{j}"] = j

    docs.append(doc)

import marqo

mq = marqo.Client("http://localhost:8882")

mq.delete_index("test_index")
mq.create_index("test_index")

mq.index("test_index").add_documents(docs, tensor_fields=["content"], client_batch_size=32)


import time

search_times = []
for i in range(200):
    start = time.time()
    mq.index("test_index").search("hello", limit=1000)
    search_times.append(time.time() - start)

print("Average search time:", sum(search_times) / len(search_times))
print("Max search time:", max(search_times))
print("Min search time:", min(search_times))
print("Total search time:", sum(search_times))