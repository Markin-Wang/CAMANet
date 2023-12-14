import torch
import argparse
import numpy as np
import os
import random
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.model import R2GenModel
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from config_swin import get_config
from tqdm import tqdm
import json


selected_id = {'01ceb247-fa13bc0e-8819e99f-9df1e9e8-bba88b3d',
 '01eaece3-70d48ee8-709d04c6-967fa1f4-a486c1fb',
 '0beab5cd-dd1bb454-0df993cf-f3c0ae3d-8f0e0c27',
 '0d41d944-b75b4101-f204d112-11fcfa1c-96d2169d',
 '0d6db000-b7832a09-4e80e472-89242ef5-20701513',
 '0f257273-0fa8c76f-737b4a98-eedda2aa-44d82e39',
 '0f513599-eb6bddc9-4306d15d-46c7c0c2-a3c6c854',
 '12d4cda1-a51a4015-46e05368-b984cb4f-10b1be5c',
 '13e67075-19ffe93c-e24d6601-d1d92120-f69369f2',
 '1675afce-31756f63-a165a417-94a2c4ab-41fa955f',
 '1955b279-efe705ba-68f22a50-df04507e-dfed9525',
 '1982caee-73cd2f56-0f1d96b7-2b66f5fc-69c0c582',
 '1a0662d4-8bee75af-c5c452a9-4b43c737-b74d27c1',
 '1a3a93cb-fcff8a20-d84a6c00-5a46ada4-2a5d437a',
 '1a5734f8-86784713-834c020a-10c75729-cff94a9b',
 '1b6cfbee-901f801d-651c11f8-2c84bb31-91883814',
 '1bc3d3de-cd13c1cd-ce13e61d-5191632c-e3ae7b5c',
 '1d1ad085-bc04d368-4062c6ff-8388f25c-c9acb192',
 '1e31fec1-1f4cbc01-4583b395-5127c6f7-43b9a7e7',
 '20926ce7-7b3d87db-c15f0a3e-556e3a75-1c54be6f',
 '22a3477a-665567e4-137b590b-c2a27bb8-d03b7d01',
 '2528f6e5-586bb3a0-e00e7283-5c594954-fe27b052',
 '279895b7-16a23c5e-1aea2909-baa62b3f-884b6f9e',
 '2a5046e4-c023b60a-61a89d1b-464d705c-e2b1eae7',
 '2b4cfcc5-c44c4f2a-8e59b25e-b354f0ac-459b3e05',
 '2d13a8b7-f90c5932-218e4fdf-056b5c2f-550c0a09',
 '3338ba8a-3a7be5a3-380128ed-7bb1359c-14e4c2d1',
 '37281a6b-d40f025d-51681f11-e078aa8f-3c6452d2',
 '378d7d48-0cfa19a3-361e40d3-6bd71394-bca64527',
 '37f7e3ca-93ef1bc3-81e615c8-a061addd-3a3b6dbf',
 '384b766e-a666fc50-5510a97f-c615a43c-1bfebe33',
 '3a00ab90-4563967d-ad46d969-ae884a78-c7f2dd2b',
 '3a031d2f-ff234adf-3d7600a9-f15a50c2-9ed90d31',
 '3fb9a12d-133444d3-6e328120-7dc4d6ef-d0b6b9fd',
 '40a34d9e-affb9a73-e6009ee7-ed1a371e-64c6a36e',
 '415f8af3-cd9c7d29-d09be965-7f210ffa-09522234',
 '425d59af-b3a07390-48699ce4-edd9cf7d-3b4faafe',
 '42cb7646-ac2acc5b-504f6247-07366b48-3d2bd573',
 '4395551b-f2717eed-fcd629df-804bb762-a356218d',
 '45aff2db-f97c8da4-6c6f992e-d40a0952-c0675aea',
 '46bd0776-78bcff84-5e4494eb-6f9c877a-a356af25',
 '48e69f6e-d7d3b831-9c09eade-bb20bccd-c9102543',
 '49e45fba-5b48f519-adb35266-68939cbb-dfda8e0f',
 '4a11826b-f6d01af0-18890057-960c5a8c-f24fc5f0',
 '4d91911d-7ed6ea7f-18ae148c-fb6fdc45-798771a7',
 '4e2ae929-40713138-9d6a757e-deeed0b1-8062cd72',
 '4e44e0c6-f6bbfa6d-36e48830-791d6141-78bb36e6',
 '4eab5702-5e51a961-a59e4e84-b5aa758f-4e367b89',
 '50ce474f-a6c1b7fd-18d97f9e-98effe01-c29ad3be',
 '5337ec0a-283bf318-55060740-77ac2e55-67b5f668',
 '5cfc2922-68cd176a-e182b4c8-e74dd44c-0ea44344',
 '5e06f576-00f63575-732b3eac-a525f7d2-9355ee5f',
 '5ecd8878-ac3a84b5-6b82b286-c4e20569-9b9f5df3',
 '6029ba23-5d73e768-c1fe417b-73eb330f-9c507e77',
 '60565158-58324362-cca18ef0-bb2bc393-750737fd',
 '61ed122d-80b347e7-d2269b6b-e28fb75e-e5585f0f',
 '622257bb-496a36b2-e8d31897-1bcc260d-c1d607d2',
 '63b80213-438bb6c2-4d070fea-92d5e59e-87611ef8',
 '64c99cbe-e1457ba5-58d940df-68b406e8-2a430fdc',
 '66607c54-01766ee9-0296b1fd-b642145d-24ea1577',
 '66fece2b-2fccf418-d23f1eda-9dde45e2-d85df8da',
 '676f6524-0bac20b4-e0e1569b-3ac3e8ee-92877aa0',
 '6b93ec0b-b35a1d19-cbcefb65-297d04fe-ca31986d',
 '704b81fb-eb6b3580-0bf2d329-f5aa33e7-5e85c2ae',
 '719206c4-ade9b6c1-79fda2c7-c9cf7be4-a8979a87',
 '742a919c-4e4a6e34-f49de182-4a0dafcf-8b3c101b',
 '74728f75-0a018add-11c546f2-e847b4e1-25501802',
 '77627414-f5a7090e-25aa3533-2b99b3af-0c5abf63',
 '777626de-a55fbd7d-e30f8359-db74c619-80afa62d',
 '783fc94d-12b747b1-600f2e10-c1c51d2a-97240f95',
 '789709af-ab78dbbd-bd973f37-aa5edc4c-cb7f975a',
 '7fa40636-0f1e59a3-7231587d-33eea7f2-79d6fae8',
 '823fd649-1a827456-8a52f457-41419696-3c50b072',
 '8262f308-02a47750-2bb9a31e-35cf7aad-6c5121f4',
 '879c5bd5-8fde6e6e-470c4bdb-323689b2-fac6fa7e',
 '8c7ee112-c1f78575-59746254-e217c9f2-81146a87',
 '8ed93a6c-a257c9c3-b7011ef7-9fd0fc17-8b045b94',
 '91bd4888-7f1222f4-5b4fe46d-db77d37b-077c6f19',
 '925c7815-b98af60d-65bf143d-402d7df3-91f83561',
 '92633e53-79ea5fb7-67adcc81-8c6f443e-7c201666',
 '928e66f1-87ef1b9e-0ce33e37-760d835a-a539e8b9',
 '96970f3a-0571b454-3baba4d3-45236f65-abf7a9c6',
 '9b4fdd07-1f45d8dc-4890ea49-e3f06306-639cb645',
 '9b571cea-6eac4eb9-c9721fa5-37624c30-9d753aea',
 '9b89dbe0-e7cb624a-a28136ca-4e93fa28-46f66f22',
 '9b9401ad-e590ff90-2ac696ba-9c7f78b2-661402b7',
 '9f5b44e9-6f162589-6533517c-f73c712d-9cef61a7',
 '9fbb07e2-d260dfd7-0f8132b6-c8b2cabb-6745996f',
 'a025f08e-de9dddc4-8716a1ac-899ce213-d7289c7a',
 'a0578edb-12a640ca-1ddab351-089c4d4c-00bb6f19',
 'a19573c3-98f76c03-5552fc10-4d2cb79e-bce663a8',
 'a19deddd-1fd8b1e8-1cd65322-2e4f8c1e-086650bd',
 'a238199b-93d2aa00-f4451329-26e4438c-e170ad89',
 'a31cf547-a85da812-785f9396-ec422967-38d69e1c',
 'a5bb1dd6-32ef2b29-b27f45f5-4980a5b0-34f11cf0',
 'a5d858a3-f180454b-311e1427-1b70d6f0-3d95426d',
 'a67e2e2b-c5902ccf-adf291f3-51b417af-5b71eeaa',
 'aa615bc7-e32c0c72-a1f0ee3f-0a7f4a52-5e7078c2',
 'ab5d8429-a48d1b05-af73d020-ef1f6e53-30f8ae8d',
 'ae711ffd-03ebb7b3-cc16c95e-e6f64de7-d2bf7de4',
 'b361a1e1-b9c3ab9d-c2cc8344-2903cfd8-3888d7b9',
 'b4a25932-1328eeb3-d6edac97-2f1a91ba-69790ccf',
 'b57face8-df2c3c57-2a99e6b1-4919f774-c8c3e93c',
 'b7d5d87f-d26475b8-59e5abac-b1142fa5-4071124e',
 'b91c97ed-5177ed0b-fa1759b1-28b3e6ac-e518d525',
 'ba93c845-aff601a7-a7342bac-ad387748-7af110b6',
 'be1ddefb-9327567f-aef38bd8-e918043d-91c40219',
 'c08e8ebb-14a3a1f0-0da1ea4e-1b2412fb-f2d4da54',
 'c190fb7d-da5b3a51-5f074369-736f62a6-589d6474',
 'c2b37067-62a9fdf0-0db4dea8-582680ef-32366c0c',
 'c476c50a-1f0890c2-aba98995-954a758b-7f46da68',
 'c6fdd21a-91b444bd-940aae07-50ed7fb6-f27ea087',
 'c9532e5b-e9cb7923-1d3cf2ef-05e252e8-dcf11149',
 'c9bd6dd6-c8328950-4f61c412-81766efb-2d9c193f',
 'cbba1c1b-baa08812-9bf09668-f10eec71-d6c20e98',
 'cd866aa1-0710b4d4-2c7e1783-c1afef62-1d1301b4',
 'cfec6d9d-4bc06a39-db51e654-c78ce642-16ef1ae3',
 'd021e279-fc2a15cf-aa08b3db-9b75b05d-324ffb18',
 'd165b008-6569b2ab-6899ea6b-f3f5f10e-481cc0dd',
 'd17e21ba-cf76b4d5-e90b2776-43be3667-dacf2f6f',
 'd1d8992c-f86b1ca8-23d9a4b4-b87ba230-a44fc7b2',
 'd3b0d36d-5201ca16-3476454c-0e031e78-004217a2',
 'd43be646-19f03d73-110ab467-b77f44ad-4f285803',
 'd50452d1-8652542d-f45133ab-196c1ef0-7bb886e0',
 'd714d837-b94d4724-3105ec18-ec20dde4-57c58bf4',
 'd78cb088-c3cad3f2-7a6176d6-7a4ca5df-dbe9326c',
 'd7f19d0e-f85e6043-96b8d9b9-fd64fd5b-7594b0ea',
 'd89f6431-69df909d-747f1354-8a38a37f-5835e7aa',
 'd8b6b619-9e181de2-c46adb2d-08194ead-eefd7108',
 'da9e3e67-02622466-3838d301-ca677b26-64a2bee0',
 'db019b7e-d9ed7caa-dce2242f-4d94ffd2-276acfb6',
 'db9e4471-b977972b-27adc624-e77cf1df-13e56a0c',
 'dcd6fbb9-e2ec404a-8b19713d-5379757a-105c3803',
 'dfa28d80-2c323234-0b53a9cc-fa22a300-37d9a55c',
 'e26fdf14-791d85bf-3beaee42-3ec8bcee-4a05efee',
 'e3fc5bd6-0ebd345c-dd63d96c-6844627c-1b6cf82b',
 'e5382fdb-74985bc4-2fb7ed30-c1708f5c-3f136ee4',
 'e7c6ee1e-e78f4a5f-8d06b880-0facc167-9037ed6a',
 'e9f8beb8-4ee1436c-72c497d0-1bc5a42c-e9cfb483',
 'ea1b22a8-7ee63c4a-1ad1ae64-defd894b-1a52dcac',
 'eca4fc13-1e4006db-4372cf2e-ed001e18-a7050d3e',
 'ee1b7363-7791f3b8-05250aa7-b16ae53b-f1d3e209',
 'f2075bc9-3c92d658-0f36d71a-9df38119-d2fafe13',
 'f3b42407-6b2326f3-2497e880-ce2defbd-96071f1d',
 'f4adee4b-4f00cc47-63f9ed2a-b4432064-a81ec91c',
 'f64708b2-5173902f-9397bc55-1a8502c8-8be61ec4',
 'fbecb95d-55942985-c9904dd9-66049a82-cd83c3a2',
 'fcedd2e4-64153d40-86614cb0-bae4c2c0-58975d3f'}

def main():
    # parse arguments
    args, config = parse_args()
    print(args)
    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    model = R2GenModel(args, tokenizer, logger, config)
    model.load_state_dict(torch.load(args.pretrained)['state_dict'])


    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())

    metrics = compute_scores

    model = model.cuda()

    model.eval()
    data = []
    image_ids = []
    image_labels = []
    image_prediction = []
    with torch.no_grad():
        records = []
        test_gts, test_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(tqdm(test_dataloader)):
            # vis_data = {}
            # vis_data['id'] = images_id
            # #vis_data['img'] = images
            # vis_data['labels'] = labels
            #
            # print(images_id)
            image_labels.append(labels)
            image_ids.extend(images_id)

            images, reports_ids, reports_masks, labels = images.cuda(), reports_ids.cuda(), \
                                                          reports_masks.cuda(), labels.cuda()
            # #if images_id[0] != 'data/mimic_cxr/images/p10/p10402372/s51966612/8797515b-595dfac0-77013a06-226b52bd-65681bf2.jpg':
            # #    continue
            # #print('000', reports_ids, reports_ids.shape)
            _, _, logits = model(images, labels=labels, mode='sample')

            image_prediction.append(logits.detach().cpu())

        #vis_data['records'] = records

    #image_ids = torch.cat(image_ids, dim=1)
    image_labels = torch.cat(image_labels, dim=0)
    image_prediction = torch.cat(image_prediction, dim=0)
    print(image_prediction.shape,len(image_ids),image_labels.shape)
    data = {'ids':image_ids,'labels':image_labels, 'pred':image_prediction}
    # f = open('mimic_prediction_our03.json', 'w', encoding='utf-8')
    # json.dump(records, f, indent=1)
    # f.close()
    torch.save(data, 'iu_cls_data/test_data.pth')
    #torch.save([tokenizer.idx2token, tokenizer.token2idx], os.path.join('visualizations','vis', args.dataset_name+'token_map.pth'))







if __name__ == '__main__':
    main()
