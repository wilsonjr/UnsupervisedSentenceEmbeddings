from typing import Any, List

from datetime import datetime

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments


class TokenizedDataset:
    def __init__(self, sentences: List, tokenizer: Any, max_length: int, cache_tokenization: bool=False) -> None:
        self.tokenizer = tokenizer 
        self.sentences = sentences 
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item: str) -> Any:
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentence[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


class MaskedLanguageModeling:

    def __init__(self, model_name: str, mlm_probability: float=0.15) -> None:


        self.output_dir = "output/mlm_{}-{}".format(model_name.replace("/", "_"),  
                                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.data_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability)

    def set_datasets(self, train_path: str, dev_path: str) -> Any:


        self.training_dataset = []
        self.development_dataset = []

        with open(train_path, 'r', encoding='utf8') as f_train:
            for line in f_train:
                line = line.strip()
                if len(line) >= 10:
                    self.training_dataset.append(line)

        with open(dev_path, 'r', encoding='utf8') as f_dev:
            for line in f_dev:
                line = line.strip()
                if len(line) >= 10:
                    self.development_dataset.append(line)

        return self


    def train(self, epochs: int, batch_size: int, info_steps: int) -> Any:

        if not self.training_dataset or len(self.training_dataset) == 0:
            raise Exception("Please, provide training and development datasets using .set_datasets()")

        max_length = 128


        train_dataset = TokenizedDataset(self.training_dataset, self.tokenizer, max_length)
        dev_dataset = TokenizedDataset(self.development_dataset, self.tokenizer, max_length, cache_tokenization=False) if len(self.development_dataset) > 0 else None
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            evaluation_strategy="steps" if self.development_dataset is not None else "no",
            per_device_train_batch_size=batch_size,
            eval_steps=info_steps,
            save_steps=info_steps,
            logging_steps=info_steps,
            save_total_limit=1,
            prediction_loss_only=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset
        )
        
        trainer.train()

        return self

    def save(self) -> Any:
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        return self



    



