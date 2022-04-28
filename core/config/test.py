import json



with open(r'DesktopApp\core\config\cfg.json', 'r') as f:
    cfg = json.load(f)


print(cfg['common_config'])



# f = open('data.json')
 

# data = json.load(f)
 

# for i in data['emp_details']:
#     print(i)
 
# # Closing file
# f.close()