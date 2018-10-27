from datiosynthesizer import describer, generator
import distributed

if __name__ == "__main__":
    #describer.init_describer(attribute_to_datatype={'ssn':'SocialSecurityNumber','age':'Integer','education':'String',
    #                                                        'marital-status':'String','relationship':'String','sex':'String','income':'String'},
    #                         attribute_to_is_categorical=['education','marital-status','relationship','sex','income'],
    #                         attribute_to_is_candidate_key = ['ssn'])

    #dd, data = describer.correlated_mode('data/adult_ssn.csv')
    #describer.save_dataset_description_to_file(dd,'out/correlated_attribute_mode/adult_ssn.json')
    file_descriptor = 'out/independent_attribute_mode/adult_ssn.json'
    total_rows = 100
    total_chunks = 10
    file_output = 'out/random_mode/adult_ssn_synthetic'
    desc = generator.init_generator(file_descriptor,total_rows,total_chunks,file_output)
    delayed = generator.generate_random(desc)
    print(delayed)
