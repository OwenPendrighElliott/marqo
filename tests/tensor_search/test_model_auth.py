"""todos: host a public HF-based CLIP (non-OpenCLIP) model so that we can use it for mocks and tests

multiprocessing should be tested manually -problem with mocking (deadlock esque)
"""
from marqo.s2_inference.random_utils import Random
from marqo.s2_inference.s2_inference import _convert_vectorized_output
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
from marqo.errors import InvalidArgError, IndexNotFoundError, BadRequestError
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.model_downloading.from_s3 import get_s3_model_absolute_cache_path
from marqo.tensor_search.models.external_apis.s3 import S3Location
from unittest import mock
import unittest
import os
from marqo.errors import BadRequestError, ModelNotInCacheError
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity


def fake_vectorise(*args, **_kwargs):
    random_model = Random(model_name='blah', embedding_dim=512)
    return _convert_vectorized_output(random_model.encode(_kwargs['content']))

def _delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass
    
    
def _get_base_index_settings():
    return {
        "index_defaults": {
            "treat_urls_and_pointers_as_images": True,
            "model": 'my_model',
            "normalize_embeddings": True,
            # notice model properties aren't here. Each test has to add it
        }
    }

class TestModelAuthLoadedS3(MarqoTestCase):
    """loads an s3 model loaded index, for tests that don't need to redownload
    the model each time """

    model_abs_path = None
    fake_access_key_id = '12345'
    fake_secret_key = 'this-is-a-secret'
    index_name_1 = "test-model-auth-index-1"
    s3_object_key = 'path/to/your/secret_model.pt'
    s3_bucket = 'your-bucket-name'
    custom_model_name = 'my_model'
    device='cpu'

    @classmethod
    def setUpClass(cls) -> None:
        """Simulates downloading a model from a private and using it in an
        add docs call
        """
        super().setUpClass()
        
        cls.endpoint = cls.authorized_url
        cls.generic_header = {"Content-type": "application/json"}
        
        try:
            tensor_search.delete_index(config=cls.config, index_name=cls.index_name_1)
        except IndexNotFoundError as s:
            pass

        cls.model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=cls.s3_object_key,
                Bucket=cls.s3_bucket
            ))
        _delete_file(cls.model_abs_path)

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": cls.s3_bucket,
                    "Key": cls.s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=cls.config, index_name=cls.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        # file should not yet exist:
        assert not os.path.isfile(cls.model_abs_path)

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
            # Call the function that uses the generate_presigned_url method
            res = tensor_search.add_documents(config=cls.config, add_docs_params=AddDocsParams(
                index_name=cls.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                model_auth=ModelAuth(
                    s3=S3Auth(aws_access_key_id=cls.fake_access_key_id, aws_secret_access_key=cls.fake_secret_key))
            ))
            assert not res['errors']

        # now the file exists
        assert os.path.isfile(cls.model_abs_path)

        mock_s3_client.generate_presigned_url.assert_called_with(
            'get_object',
            Params={'Bucket': 'your-bucket-name', 'Key': cls.s3_object_key}
        )
        mock_boto3_client.assert_called_once_with(
            's3',
            aws_access_key_id=cls.fake_access_key_id,
            aws_secret_access_key=cls.fake_secret_key,
            aws_session_token=None
        )
        
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        _delete_file(cls.model_abs_path)
        tensor_search.eject_model(model_name=cls.custom_model_name, device=cls.device)

    def test_after_downloading_auth_doesnt_matter(self):
        """on this instance, at least"""
        res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
        ))
        assert not res['errors']

    def test_after_downloading_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
            ))
            assert not res['errors']
            mock_req.assert_not_called()
        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])

    def test_after_downloading_search_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.search(config=self.config,
                index_name=self.index_name_1, text='hi'
            )
            assert 'hits' in res
            mock_req.assert_not_called()

        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])

class TestModelAuth(MarqoTestCase):

    device = 'cpu'

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_model_auth_hf(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.pt'

        mock_open_clip_creat_model = mock.MagicMock()

        with unittest.mock.patch('open_clip.create_model_and_transforms', mock_open_clip_creat_model):
            with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                try:
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                        model_auth=ModelAuth(hf=HfAuth(token=hf_token))))
                except BadRequestError as e:
                    # bad request due to no models actually being loaded
                    print(e)
                    pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object
        )

        # is the open clip model being loaded with the expected args?
        called_with_expected_args = any(
            call.kwargs.get("pretrained") == "cache/path/to/model.pt"
            and call.kwargs.get("model_name") == "ViT-B/32"
            for call in mock_open_clip_creat_model.call_args_list
        )
        assert len(mock_open_clip_creat_model.call_args_list) == 1
        assert called_with_expected_args, "Expected call not found"

    def test_model_auth_s3_search(self):
        """The other test load from add_docs, we have to make sure it works for
         search"""

        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=s3_object_key,
                Bucket=s3_bucket
        ))
        _delete_file(model_abs_path)

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'
        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        # file should not yet exist:
        assert not os.path.isfile(model_abs_path)

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client)  as mock_boto3_client:
            res = tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=ModelAuth(s3=S3Auth(aws_access_key_id=fake_access_key_id, aws_secret_access_key=fake_secret_key))
            )

        assert os.path.isfile(model_abs_path)

        mock_s3_client.generate_presigned_url.assert_called_with(
            'get_object',
            Params={'Bucket': 'your-bucket-name', 'Key': s3_object_key}
        )
        mock_boto3_client.assert_called_once_with(
            's3',
            aws_access_key_id=fake_access_key_id,
            aws_secret_access_key=fake_secret_key,
            aws_session_token=None
        )
        _delete_file(model_abs_path)

    def test_model_auth_hf_search(self):
        """The other test focused on add_docs. This focuses on search
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.pt'

        mock_open_clip_creat_model = mock.MagicMock()

        with unittest.mock.patch('open_clip.create_model_and_transforms', mock_open_clip_creat_model):
            with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                try:
                    res = tensor_search.search(
                        config=self.config, text='hello', index_name=self.index_name_1,
                        model_auth=ModelAuth(hf=HfAuth(token=hf_token)))
                except BadRequestError:
                    # bad request due to no models actually being loaded
                    pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object
        )

        # is the open clip model being loaded with the expected args?
        called_with_expected_args = any(
            call.kwargs.get("pretrained") == "cache/path/to/model.pt"
            and call.kwargs.get("model_name") == "ViT-B/32"
            for call in mock_open_clip_creat_model.call_args_list
        )
        assert len(mock_open_clip_creat_model.call_args_list) == 1
        assert called_with_expected_args, "Expected call not found"

    def test_model_auth_mismatch_param_s3_ix(self):
        """There isn't validation for the hf because users may download public models this way"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        hf_token = 'hf_secret_token'

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm:
                tensor_search.search(
                    config=self.config, text='hello', index_name=self.index_name_1,
                    model_auth=ModelAuth(hf=HfAuth(token=hf_token)))

                self.assertIn("s3 authorisation information is required", str(cm.exception))

    def test_model_loads_from_all_add_docs_derivatives(self):
        """Does it work from add_docs, add_docs orchestrator and add_documents_mp?
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for add_docs_method, kwargs in [
                (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(BadRequestError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        add_docs_method(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=self.index_name_1,
                                model_auth=ModelAuth(s3=S3Auth(
                                    aws_access_key_id=fake_access_key_id,
                                    aws_secret_access_key=fake_secret_key)),
                                auto_refresh=True,
                                docs=[{f'Title': "something {i} good"} for i in range(20)]
                            ),
                            **kwargs
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url='https://some_non_existent_model.pt', cache_dir=None, cache_file_name='secret_model.pt')
            mock_s3_client.generate_presigned_url.assert_called_with(
                'get_object',
                Params={'Bucket': 'your-bucket-name', 'Key': s3_object_key}
            )
            mock_boto3_client.assert_called_once_with(
                's3',
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key,
                aws_session_token=None
            )
            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_model_loads_from_multi_search(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        random_model = Random(model_name='blah', embedding_dim=512)

        try:
            tensor_search.eject_model(model_name='my_model', device=self.device)
        except ModelNotInCacheError:
            pass
        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

        with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                 side_effect=fake_vectorise) as mock_vectorise:
            model_auth = ModelAuth(
                s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
            )
            res = tensor_search.search(
                index_name=self.index_name_1,
                config=self.config,
                model_auth=model_auth,
                text={
                    (f"https://raw.githubusercontent.com/marqo-ai/"
                     f"marqo-api-tests/mainline/assets/ai_hippo_realistic.png"): 0.3,
                    'my text': -1.3
                },
            )
            assert 'hits' in res
            mock_vectorise.assert_called()
            assert len(mock_vectorise.call_args_list) > 0
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

    def test_model_loads_from_multimodal_combination(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        random_model = Random(model_name='blah', embedding_dim=512)


        for add_docs_method, kwargs in [
            (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            (tensor_search.add_documents, {})
        ]:
            try:
                tensor_search.eject_model(model_name='my_model', device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise', side_effect=fake_vectorise) as mock_vectorise:
                model_auth = ModelAuth(
                    s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
                )
                res = add_docs_method(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        model_auth=model_auth,
                        auto_refresh=True,
                        docs=[{
                            'my_combination_field': {
                                'my_image': f"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                                'some_text': f"my text {i}"}} for i in range(20)],
                        mappings={
                            "my_combination_field": {
                                "type": "multimodal_combination",
                                "weights": {
                                    "my_image": 0.5,
                                    "some_text": 0.5
                                }
                            }
                        }
                    ),
                    **kwargs
                )
                if isinstance(res, list):
                    assert all([not batch_res ['errors'] for batch_res in res])
                else:
                    assert not res['errors']
                mock_vectorise.assert_called()
                for _args, _kwargs in mock_vectorise.call_args_list:
                    assert _kwargs['model_properties']['model_location'] == {
                        "s3": {
                            "Bucket": s3_bucket,
                            "Key": s3_object_key,
                        },
                        "auth_required": True
                    }
                    assert _kwargs['model_auth'] == model_auth

    def test_no_creds_error(self):
        """in s3, if there aren't creds"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        hf_token = 'hf_secret_token'

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm:
                tensor_search.search(
                    config=self.config, text='hello', index_name=self.index_name_1,
                )
                self.assertIn("s3 authorisation information is required", str(cm.exception))


        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm2:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True,
                        docs=[{'title': 'blah blah'}]
                    )
                )
            self.assertIn("s3 authorisation information is required", str(cm2.exception))

    def test_bad_creds_error_s3(self):
        """in s3 if creds aren't valid. Ensure a helpful error"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm2.exception))

    def test_non_existent_hf_location(self):
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )

        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bad_creds_error_hf(self):
        """the model and repo do exist, but creds are bad. raises the same type of error
        as the previous one. """
        hf_object = "dummy_model.pt"
        hf_repo_name = "Marqo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bulk_search(self):
        """Does it work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(InvalidArgError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url='https://some_non_existent_model.pt', cache_dir=None, cache_file_name='secret_model.pt')
            mock_s3_client.generate_presigned_url.assert_called_with(
                'get_object',
                Params={'Bucket': 'your-bucket-name', 'Key': s3_object_key}
            )
            mock_boto3_client.assert_called_once_with(
                's3',
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key,
                aws_session_token=None
            )

            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_bulk_search_vectorise(self):
        """are the calls to vectorise expected? work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                     side_effect=fake_vectorise) as mock_vectorise:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_vectorise.assert_called()
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

            mock_vectorise.reset_mock()

    def test_lexical_with_auth(self):
        """should just skip"""

    def test_public_s3_no_auth(self):
        """
        TODO
        """

    def test_public_hf_no_auth(self):
        """
        TODO
        """

    def test_open_clip_reg_clip(self):
        """both normal and open clip
        TODO: normal CLIP
        """








