from datiosynthesizer import describer

if __name__ == "__main__":
    #client = Client()
    describer.init_describer(attribute_to_datatype={'ssn':'SocialSecurityNumber','age':'Integer','education':'String',
                                                            'marital-status':'String','relationship':'String','sex':'String','income':'String'},
                             attribute_to_is_categorical=['education','marital-status','relationship','sex','income'],
                             attribute_to_is_candidate_key = ['ssn'])

    desc, _ = describer.correlated_mode('data/adult_ssn.csv')
    print(desc)
    describer.save_dataset_description_to_file(desc, 'out/correlated_attribute_mode/adult_ssn.json')
