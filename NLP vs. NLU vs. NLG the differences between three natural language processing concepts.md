## NLP vs. NLU vs. NLG: the differences between three natural language processing concepts

### NLU

Natural language understanding is a subset of natural language processing, 使用语法语义去分析文本和演讲去决定句子的意思。语法决定句子的结构，语义决定句子的意思。

NLU也就是一种指定单词和句子关系的数据结构。需要区分同音不同意的单词。或者说是一词多义。

1. Alice is swimming against the current.
2. The current version of the report is in the folder.

In the first sentence, the word, current is a noun. The verb that precedes it, swimming, provides additional context to the reader, allowing us to conclude that we are referring to the flow of water in the ocean. The second sentence uses the word current, but as an adjective. The noun it describes, version, denotes multiple iterations of a report, enabling us to determine that we are referring to the most up-to-date status of a file.

These approaches are also commonly used in data mining to understand consumer attitudes.

### NLG

Natural language generation is another subset of natural language processing

While natural language understanding focuses on computer reading comprehension, natural language generation enables computers to write. NLG is the process of producing a human language text response based on some data input. This text can also be converted into a speech format through text-to-speech services.

As with NLU, NLG applications need to consider language rules based on morphology, lexicons, syntax and semantics to make choices on how to phrase responses appropriately. They tackle this in three stages:

- Text planning: During this stage, general content is formulated and ordered in a logical manner.
- Sentence planning: This stage considers punctuation and text flow, breaking out content into paragraphs and sentences and incorporating pronouns or conjunctions where appropriate.
- Realization: This stage accounts for grammatical accuracy, ensuring that rules around punctation and conjugations are followed. For example, the past tense of the verb *run* is *ran*, not runned.

### NLP vs NLU vs. NLG summary

- **Natural language processing (NLP)** seeks to convert unstructured language data into a structured data format to enable machines to understand speech and text and formulate relevant, contextual responses. Its subtopics include natural language processing and natural language generation.
- **Natural language understanding (NLU)**focuses on machine reading comprehension through grammar and context, enabling it to determine the intended meaning of a sentence.
- **Natural language generation (NLG)** focuses on text generation, or the construction of text in English or other languages, by a machine and based on a given dataset.