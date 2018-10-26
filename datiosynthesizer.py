from datiosynthesizer import describer, generator
import distributed

if __name__ == "__main__":
    #describer.init_describer(attribute_to_datatype={'ssn':'SocialSecurityNumber','age':'Integer','education':'String',
    #                                                        'marital-status':'String','relationship':'String','sex':'String','income':'String'},
    #                         attribute_to_is_categorical=['education','marital-status','relationship','sex','income'],
    #                         attribute_to_is_candidate_key = ['ssn'])

    #dd, data = describer.correlated_mode('data/adult_ssn.csv')
    #describer.save_dataset_description_to_file(dd,'out/correlated_attribute_mode/adult_ssn.json')

    desc = generator.init_generator('out/correlated_attribute_mode/adult_ssn.json',100,10,'out/correlated_attribute_mode/adult_ssn_synthetic.parquet')
    delayed = generator.generate_random(desc)
    print(delayed)
