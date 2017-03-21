import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Emmitt on 3/3/2017.
 */

public class ReadFile {
    //Put all file names into a list
    public List<String> getAllFileName(String dir_path) {
        File folder = new File(dir_path);
        List<String> fileNames=new ArrayList<String>();
        if(!folder.exists()){
            System.out.println("No such directory as "+dir_path);
            return fileNames;
        }
        File[] listOfFiles = folder.listFiles();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                fileNames.add(listOfFiles[i].getName());
            }
        }
        return fileNames;
    }
    //Get the contents of file, return it as a string
    public String readFile(String filename) throws IOException {
        String everything="";
        BufferedReader br = new BufferedReader(new FileReader(filename));
        try {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                line = br.readLine();
            }
            everything = sb.toString();
            } finally {
                br.close();
            }
            return everything;
    }
}
