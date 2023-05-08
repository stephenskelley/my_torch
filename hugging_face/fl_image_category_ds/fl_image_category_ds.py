# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset class for fl_image_category_ds."""

import datasets
import pandas
from datasets.tasks import ImageClassification
from algorithm_library.algorithm import Algorithm
import local_secrets as secrets
import urllib.request

_BASE_URL = ''
_METADATA_URLS = {
    'train': '',
    'test': '',
}
_HOMEPAGE = ''
_DESCRIPTION = (
    "To fine tune image recognition model for FL product categories"
    ""
)
_CITATION = """\
"""
_LICENSE = """\
LICENSE AGREEMENT
=================
"""
_NAMES = ['Bottom', 'Top', 'Bra', 'Outfits', 'Accessory', 'Shoes', 'Savage', 'Underwear', 'Jacket']
_IMAGES_DIR = ''


class FLImageCategory(datasets.GeneratorBasedBuilder):

    def __init__(self, *args, **kwargs):
        self.algo = Algorithm(secrets)
        self.df = self.get_image_df()
        super(FLImageCategory, self).__init__(*args, **kwargs)

    def get_image_df(self):
        df = self.algo.snowflake_query_to_pandas('select * from work.skelley.SKU_IMAGES sample (10)')
        df.loc[df.product_category == 'Sock/Underwear', 'product_category'] = 'Underwear'
        df.loc[df.product_category == 'Bottoms', 'product_category'] = 'Bottom'
        df.loc[df.product_category == 'Legging', 'product_category'] = 'Bottom'
        df.loc[df.product_category == 'Accessories', 'product_category'] = 'Accessory'
        df.loc[df.product_category == 'Accessories', 'product_category'] = 'Accessory'
        df = df.rename(columns={'image_link': 'additional_image_link_0'})
        df = df[df.product_category.isin(_NAMES)]
        return df

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=_NAMES),
                    'sku': datasets.Value(dtype='string', id=None),
                    'mpid': datasets.Value(dtype='int32', id=None),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        #archive_path = dl_manager.download(_BASE_URL)
        #split_metadata_paths = dl_manager.download(_METADATA_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": 'train',
                    "metadata_path": '',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images": 'validation',
                    "metadata_path": '',
                },
            ),
        ]

    def _generate_examples(self, images, metadata_path):
        """Generate images and labels for splits."""
        if images == 'validation':
            df = pandas.DataFrame()
        else:
            df = self.df
        for index, row in df.iterrows():
            for i in range(6):
                file_path = row[f'additional_image_link_{i}']
                try:
                    response = urllib.request.urlopen(urllib.request.Request(file_path))
                except urllib.error.URLError as e:
                    continue
                if response.code != 200:
                    continue

                yield file_path, {
                    "image": {"path": file_path, "bytes": response.read()},
                    "label": row['product_category'],
                    'sku': row['sku'],
                    'mpid': row['master_product_id']
                }

        # with open(metadata_path, encoding="utf-8") as f:
        #     files_to_keep = set(f.read().split("\n"))
        # for file_path, file_obj in images:
        #     if file_path.startswith(_IMAGES_DIR):
        #         if file_path[len(_IMAGES_DIR) : -len(".jpg")] in files_to_keep:
        #             label = file_path.split("/")[2]
        #             yield file_path, {
        #                 "image": {"path": file_path, "bytes": file_obj.read()},
        #                 "label": label,
        #             }