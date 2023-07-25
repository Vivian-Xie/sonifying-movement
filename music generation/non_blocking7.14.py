import threading
import queue
import mido
import pygame
import pygame.midi
from socket import *
import numpy as np
import torch
import pretty_midi as pm
import mido
import os
import time
from ec2vae.model import EC2VAE
import sys
if not os.path.exists('./demo'):
    os.mkdir('./demo')

# 音乐播放
def play_music_thread(music_queue):
    pygame.mixer.init()
    while True:
        file_path = music_queue.get('./demo/ec2vae-new-chord.mid')
        if file_path is None:
            break
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        '''
        # 加载MIDI文件
        mid = mido.MidiFile(file_path)
        # 创建一个输出端口
        output = pygame.midi.Output(0)
        
        # 遍历MIDI文件中的每个track
        for i, track in enumerate(mid.tracks):
            # 播放每个track中的MIDI事件
            for msg in track:
                # 如果是Note On或Note Off事件,则发送到输出端口
                if msg.type == 'note_on' or msg.type == 'note_off':
                    output.note_on(msg.note, velocity=msg.velocity, channel=msg.channel)
                    pygame.time.wait(int(msg.time * 1000))
                    output.note_off(msg.note, velocity=msg.velocity, channel=msg.channel)
        # 关闭输出端口
        output.close()'''
    # 退出pygame
    pygame.quit()
    
#用emotion处理音乐
def for_socket(emotion,num):
    # initialize the model
    ec2vae_model = EC2VAE.init_model()
    # load model parameter
    ec2vae_param_path = './ec2vae/model_param/ec2vae-v1.pt'
    ec2vae_model.load_model(ec2vae_param_path)
    # x1: "From the new world" melody
    file=open(file='notes.txt',mode='r')
    data=file.readlines()
    #for i in range(len(data)):
        #data[i]=data[i].rstrip('\n')
    noteslist=[int(i.rstrip('\n')) for i in data][32*num:32*(num+1)]
    if len(noteslist)==0:
        sys.exit()
    print(noteslist)
    x1 = np.array(noteslist)
    #x1 = np.array([64, 128, 128, 67, 67, 128, 128, 128, 64, 128, 128, 62, 60, 128, 128, 128,62, 128, 128, 64, 67, 128, 128, 64, 62, 128, 128, 128, 129, 129, 129, 129])

    # x2: C4, sixteenth notes.
    x2 = np.array([60,128] * 16)    #可以用128延长音来改变rhythm

    def note_array_to_onehot(note_array):
        pr = np.zeros((len(note_array), 130))
        pr[np.arange(0, len(note_array)), note_array.astype(int)] = 1.
        return pr


    pr1 = note_array_to_onehot(x1)
    pr2 = note_array_to_onehot(x2)
    #plt.imshow(pr1, aspect='auto')
    #plt.title('Display pr1')
    #plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # to pytorch tensor
    pr1 = torch.from_numpy(pr1)
    # to float32
    pr1 = pr1.float()  
    # to device (if to cpu, the operation can be omitted.)
    pr1 = pr1.to(device)
    # unsqueeze the batch dim
    pr1 = pr1.unsqueeze(0)
    # Concert pr2 similarly
    pr2 = torch.from_numpy(pr2).float().to(device).unsqueeze(0)


    amin=[1,0,0,0,1,0,0,0,0,1,0,0]
    amaj=[0,1,0,0,1,0,0,0,0,1,0,0]
    dmin=[0,0,1,0,0,1,0,0,0,1,0,0]
    dmaj=[0,0,1,0,0,0,1,0,0,1,0,0]
    emin=[0,0,0,0,1,0,0,1,0,0,0,1]
    emaj=[0,0,0,0,1,0,0,0,1,0,0,1]
    cmin=[1,0,0,1,0,0,0,1,0,0,0,0]
    cmaj=[1,0,0,0,1,0,0,1,0,0,0,0]
    gmin=[0,0,1,0,0,0,0,1,0,0,1,0]
    gmaj=[0,0,1,0,0,0,0,1,0,0,0,1]
    fmin=[1,0,0,0,0,1,0,0,1,0,0,0]
    fmaj=[1,0,0,0,0,1,0,0,0,1,0,0]
    bmaj=[0,0,0,1,0,0,1,0,0,0,0,1]
    bmin=[0,0,1,0,0,0,1,0,0,0,0,1]
    dguf=[0,1,0,0,0,0,1,0,0,0,1,0]
    deud=[0,0,0,1,0,0,1,0,0,0,1,0]
    dduc=[0,1,0,0,0,1,0,0,1,0,0,0]
    dema=[0,0,0,1,0,0,0,1,0,0,1,0]
    dama=[1,0,0,1,0,0,0,0,1,0,0,0]
    ucmi=[0,1,0,0,1,0,0,0,1,0,0,0]
    dbua=[0,1,0,0,0,1,0,0,0,0,1,0]
    ufmi=[0,1,0,0,0,0,1,0,0,1,0,0]
    ugda=[0,0,0,1,0,0,0,0,1,0,0,1]
    dbma=[0,0,1,0,0,1,0,0,0,0,1,0]

    chord=dbma
    if emotion=='angry':
        chord=amin
    if emotion=='disgust':
        chord=amaj
    if emotion=='fear':
        chord=dmin
    if emotion=='happy':
        chord=dmaj
    if emotion=='sad':
        chord=emin
    if emotion=='surprise':
        chord=emaj
    if emotion=='neutral':
        chord=cmin
    
    c1 = np.array([amin] * 8 + [dmin] * 8+[emin]*16)
    #c1 = np.array([cmaj] * 8 + [fmaj] * 8 + [gmaj]*16)
    c2 = np.array([chord]*32)
    #c2min  np.array([emin]*32)
    #c2maj = np.array([fmaj]*32)
    # no chord
    c3 = np.zeros((32, 12))
    

    c1 = torch.from_numpy(c1).float().to(device).unsqueeze(0)
    c2 = torch.from_numpy(c2).float().to(device).unsqueeze(0)
    #c2min = torch.from_numpy(c2min).float().to(device).unsqueeze(0)
    #c2maj = torch.from_numpy(c2maj).float().to(device).unsqueeze(0)
    c3 = torch.from_numpy(c3).float().to(device).unsqueeze(0)
    print(pr1)
    # encode melody 1 and chord C-G
    zp1, zr1 = ec2vae_model.encoder(pr1, c1)
    # encode melody 2 and "no chord"
    zp2, zr2 = ec2vae_model.encoder(pr2, c3)

    pred_recon = ec2vae_model.decoder(zp1, zr1, c1)
    pred_new_rhythm = ec2vae_model.decoder(zp1, zr2, c1)
    pred_new_chord = ec2vae_model.decoder(zp1, zr1, c2)
    #pred_new_chordmaj = ec2vae_model.decoder(zp1, zr1, c2maj)
    #pred_new_chordmin = ec2vae_model.decoder(zp1, zr1, c2min)

    out_recon = pred_recon.squeeze(0).cpu().numpy()
    out_new_rhythm = pred_new_rhythm.squeeze(0).cpu().numpy()
    out_new_chord = pred_new_chord.squeeze(0).cpu().numpy()
    #out_new_chordmaj = pred_new_chordmaj.squeeze(0).cpu().numpy()
    #out_new_chordmin = pred_new_chordmin.squeeze(0).cpu().numpy()

    out_new_rhythm.shape
    print(out_new_chord)
    '''
    file=open(file='/Users/yiruzhou/Documents/durf/icm-deep-music-generation-main/notes.txt',mode='w+')
    for i in out_new_chord:
        file.write(str(i)+'\n')
    file.close()
    '''
    notes_recon = ec2vae_model.__class__.note_array_to_notes(out_recon, bpm=140, start=0.)
    notes_new_rhythm = ec2vae_model.__class__.note_array_to_notes(out_new_rhythm, bpm=140, start=0.)
    notes_new_chord = ec2vae_model.__class__.note_array_to_notes(out_new_chord, bpm=140, start=0.)

    notes_c1 = ec2vae_model.__class__.chord_to_notes(c1.squeeze(0).cpu().numpy(), 140, 0)
    notes_c2 = ec2vae_model.__class__.chord_to_notes(c2.squeeze(0).cpu().numpy(), 140, 0)
    #notes_c2maj = ec2vae_model.__class__.chord_to_notes(c2maj.squeeze(0).cpu().numpy(), 140, 0)
    #notes_c2min = ec2vae_model.__class__.chord_to_notes(c2min.squeeze(0).cpu().numpy(), 140, 0)

    def generate_midi_with_melody_chord(fn, mel_notes, c_notes):
        midi = pm.PrettyMIDI()
        ins1 = pm.Instrument(0)
        ins1.notes = mel_notes
        ins2 = pm.Instrument(0)
        ins2.notes = c_notes
        midi.instruments.append(ins1)
        midi.instruments.append(ins2)
        midi.write(fn)
        
    generate_midi_with_melody_chord('./demo/ec2vae-recon.mid', notes_recon, notes_c1)
    generate_midi_with_melody_chord('./demo/ec2vae-new-rhythm.mid', notes_new_rhythm, notes_c1)
    generate_midi_with_melody_chord('./demo/ec2vae-new-chord.mid', notes_new_chord, notes_c2)

    def concatenate_midi_files(file1, file2, output_file):
        midi1 = mido.MidiFile(file1)
        midi2 = mido.MidiFile(file2)
        merged_midi = mido.MidiFile(ticks_per_beat=min(midi1.ticks_per_beat, midi2.ticks_per_beat))
        track0_merged = mido.MidiTrack()
        track1_merged = mido.MidiTrack()
        track2_merged = mido.MidiTrack()
        for msg in midi1.tracks[0]:
            track2_merged.append(msg.copy())
        for msg in midi2.tracks[0]:
            track2_merged.append(msg.copy())
        merged_midi.tracks.append(track0_merged)
        for msg in midi1.tracks[1]:
            track2_merged.append(msg.copy())
        for msg in midi2.tracks[1]:
            track2_merged.append(msg.copy())
        merged_midi.tracks.append(track1_merged)
        for msg in midi1.tracks[2]:
            track1_merged.append(msg.copy())
        for msg in midi2.tracks[2]:
            track1_merged.append(msg.copy())
        merged_midi.tracks.append(track2_merged)
        

        merged_midi.save(output_file)
        
    concatenate_midi_files('./demo/ec2vae-new-chord.mid','./demo/origin.mid','./demo/origin.mid')

# 信息处理
def process_info_thread(music_queue, info_queue):
    num=0
    while True:
        info = info_queue.get()
        if info is None:
            break
        for_socket(info,num)
        music_queue.put("./demo/ec2vae-new-chord.mid")
        num+=1
        
music_queue = queue.Queue()
info_queue = queue.Queue()

music_thread = threading.Thread(target=play_music_thread, args=(music_queue,))
info_thread = threading.Thread(target=process_info_thread, args=(music_queue, info_queue))

# 启动线程
music_thread.start()
info_thread.start()
music_queue.put("./demo/ec2vae-new-chord.mid")
#此处为单机调试
while True:
    info = input("请输入信息：")
    if info!='':
        info_queue.put(info)
    if info == "stop":
        break
'''
#传送emotion
print("client is activated")
host_name="192.168.99.132"
port_num=1200
clientSocket=socket(AF_INET,SOCK_STREAM)
#print('234234')
clientSocket.connect((host_name,port_num))
#print('-----')
while True:
    info=clientSocket.recv(1024).decode()
    print('123123')
    if info!='':
        print("message from the server: "+info)
        info_queue.put(info)
    if info == "q":
        break
clientSocket.close()

'''
# 等待音乐播放线程和信息处理线程结束
music_queue.put(None)
info_queue.put(None)
music_thread.join()
info_thread.join()