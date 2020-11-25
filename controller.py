import read_data
import predict

if __name__ == '__main__':
    # set parameters
    redo_preprocessing = True
    BoW = False
    TFIDF = False
    better_TFIDF = True
    create_word_vector_model = False
    read_data.create_data_for_models(redo_preprocessing, BoW, TFIDF, better_TFIDF, create_word_vector_model)
    predict.fit_and_evaluate_models(BoW, TFIDF, better_TFIDF, create_word_vector_model)

    # test on new text
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]
    # models can be one of: BNB, LR, SVC
    # data ca be BoW, TFIDF_no_ngram, TFIDF_ngram
    predict.predict_text_sentiment('TFIDF_ngram', 'SVC', text)



