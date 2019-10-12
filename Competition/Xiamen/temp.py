    validate_target = origin_train_target[:config.SPLIT_VAL_TEST_target]
    test_target = origin_train_target[config.SPLIT_VAL_TEST_target:config.SPLIT_TEST_TRAIN_target]
    train_target = origin_train_target[config.SPLIT_TEST_TRAIN_target:]